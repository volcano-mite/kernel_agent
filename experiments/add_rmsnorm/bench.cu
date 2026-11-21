#include <cuda_bf16.h>  // cuda_bf16.h 提供 BF16 类型与转换函数。
#include <cuda_runtime.h>  // cuda_runtime.h 提供设备、内存、事件和错误检查 API。

#include <algorithm>  // algorithm 提供排序。
#include <array>  // array 提供固定大小数组。
#include <cmath>  // cmath 提供 CPU 开方和绝对值。
#include <cstdio>  // cstdio 提供日志和 CSV 输出。
#include <cstdlib>  // cstdlib 提供错误时退出进程的函数。
#include <string_view>  // 解析命令行选项。
#include <vector>  // vector 提供随 batch 调整大小的 CPU 缓冲区。

using BF16 = __nv_bfloat16;  // 为 CUDA BF16 类型起简名。
constexpr int hidden = 4096;  // 固定每行 4096 列。
constexpr int threads = 256;  // 固定每个 block 256 个线程。
constexpr int items = hidden / threads;  // 每线程 4096 / 256 = 16 个元素。

void check(cudaError_t status) {  // 定义 CUDA 状态检查函数。
    if (status != cudaSuccess) {  // 判断 API 返回值是否成功。
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));  // 将 CUDA 错误描述写到 stderr。
        std::exit(1);  // 退出码 1 表示 CUDA 调用失败。
    }
}

__global__ void add(const BF16* x, const BF16* residual, BF16* sum, int count) {  // 定义逐元素 Add kernel，sum 指向全局内存。
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 把 block 和线程索引展平成元素下标。
    if (i < count) {  // 跳过超出元素总数的线程。
        sum[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(residual[i]));  // 从全局内存读 x 和 residual，转 FP32 相加，舍入成 BF16 并写入全局内存 sum。
    }
}

template <bool fused>  // 用编译期布尔参数区分是否在同一个 kernel 内做 Add。
__device__ void normalize(const BF16* x, const BF16* residual, const BF16* weight,  // 声明设备函数的输入、残差和权重。
                          BF16* output, float* partial) {  // 接收全局输出指针及 block 内的 shared memory 指针。
    int lane = threadIdx.x;  // lane 是整个 block 内的线程编号，范围 0–255。
    int row = blockIdx.x * hidden;  // row 是当前行首元素的偏移量，由 blockIdx.x 决定。
    float values[items];  // 每线程保留 16 个 FP32 值供输出阶段复用。
    float squares = 0;  // 初始化该线程的局部平方和。
    #pragma unroll  // 请求编译器展开定长循环。
    for (int j = 0; j < items; ++j) {  // 遍历线程负责的 16 个元素。
        int i = row + lane + j * threads;  // 连续线程访问连续列。线程 t 负责 t、t+256 等列。
        float value = __bfloat162float(x[i]);  // 读全局内存并转 FP32。拆分版此处读的是 sum，融合版读的是原始 x。
        if constexpr (fused) {  // 在编译期选择是否保留 Add 分支。
            value = __bfloat162float(__float2bfloat16(value + __bfloat162float(residual[i])));  // 从全局内存读 residual，相加后先舍入 BF16，再转回 FP32，以便后续计算与拆分版对齐。
        }
        values[j] = value;  // 将当前值暂存在每线程的 values 数组。
        squares += value * value;  // 累加当前值的平方。
    }
    partial[lane] = squares;  // 把当前线程的局部平方和写进 shared memory 的独立槽位。
    __syncthreads();  // 等待全部线程完成写入，再开始读其他线程的部分和。
    for (int stride = threads / 2; stride > 0; stride /= 2) {  // 将归约跨度依次缩到 128、64，直到 1。
        if (lane < stride) {  // 只有编号小于本轮跨度的线程做加法。
            partial[lane] += partial[lane + stride];  // 读取相隔 stride 的两个部分和，把相加结果留在前半区。
        }
        __syncthreads();  // 全 block 同步，下一轮可以读取本轮结果。同步在 if 外面，因此所有线程都会经过。
    }
    float scale = rsqrtf(partial[0] / hidden + 1e-5f);  // partial[0] 是整行平方和。除以 hidden，加入 epsilon，再计算平方根的倒数。
    #pragma unroll  // 请求编译器展开输出循环。
    for (int j = 0; j < items; ++j) {  // 再遍历当前线程负责的 16 列。
        int col = lane + j * threads;  // 算出当前列号，用来索引共享权重。
        output[row + col] =  // 计算全局输出地址，对应当前行和列。
            __float2bfloat16(values[j] * scale * __bfloat162float(weight[col]));  // 从全局内存读取权重，乘上线程保留的值和缩放系数，转回 BF16 后写出。
    }
}

__global__ void rmsnorm(const BF16* sum, const BF16* weight, BF16* output) {  // 声明拆分流程的第二个 kernel，接收已算好的 sum。
    __shared__ float partial[threads];  // 为每个 block 分配 256 个 FP32 部分和槽位，共 1024 B。
    normalize<false>(sum, nullptr, weight, output, partial);  // 传 sum 作为 normalize 的输入。false specialization 会移除 residual 访问，可传 nullptr。
}

__global__ void add_rmsnorm(const BF16* x, const BF16* residual,  // 声明直接接收原始输入和残差的融合 kernel。
                            const BF16* weight, BF16* output) {  // 接收按列共享的权重和全局输出缓冲区。
    __shared__ float partial[threads];  // 分配同样大小的 shared memory。
    normalize<true>(x, residual, weight, output, partial);  // true specialization 在读取输入后立刻做 Add。
}

int main(int argc, char** argv) {  // 进入 CPU 端 benchmark。
    const std::string_view mode = argc == 2 ? argv[1] : "";  // 可选模式由一个参数指定。
    if (argc != 1 && mode != "--check-only" && mode != "--check-large") {
        std::fprintf(stderr, "Usage: %s [--check-only | --check-large]\n", argv[0]);
        return 1;
    }
    const bool check_only = !mode.empty();  // 验算模式只执行指定 batch。
    cudaDeviceProp device{};  // 初始化设备信息结构。
    check(cudaGetDeviceProperties(&device, 0));  // 查询可见设备 0。设置 CUDA_VISIBLE_DEVICES 后，0 指其可见设备列表的第一张卡。
    std::fprintf(stderr, "GPU: %s\n", device.name);  // 把 GPU 名写到 stderr，让 stdout 保持 CSV 格式。
    constexpr int repeats = 100;  // 每次计时连续提交 100 次调用。
    constexpr int trials = 7;  // 每个版本重复 7 轮。
    if (!check_only) std::puts("batch,split_us,fused_us,speedup,split_min_us,split_max_us,fused_min_us,fused_max_us");  // 输出 CSV 列名，延迟单位为微秒。
    for (int batch : {1, 16, 64, 170, 8804, 14509}) {  // 遍历六种 batch，先测小矩阵，再测大矩阵。
        if (check_only && batch != (mode == "--check-large" ? 14509 : 1)) continue;
        int count = batch * hidden;  // 计算 x、residual 和输出的元素数。
        std::vector<BF16> host_x(count), host_r(count), host_w(hidden);  // 为 CPU 输入和权重分配数组。
        for (int i = 0; i < count; ++i) {  // 逐元素初始化两个输入。
            host_x[i] = __float2bfloat16((i % 251 - 125) / 59.f);  // 用取模生成确定性 x，并转 BF16。
            host_r[i] = __float2bfloat16((i % 113 - 56) / 43.f);  // 用另一组周期和值域生成 residual，并转 BF16。
        }
        for (int i = 0; i < hidden; ++i) {  // 遍历 4096 个权重。
            host_w[i] = __float2bfloat16(0.5f + (i % 127) / 128.f);  // 给每列生成确定性 BF16 权重。
        }

        BF16 *x, *residual, *weight, *sum, *split, *fused;  // 声明设备指针，分别指向两个输入、权重、中间结果和两个输出。
        size_t bytes = count * sizeof(BF16);  // 计算每个完整张量所需字节数。
        check(cudaMalloc(&x, bytes));  // 在 GPU 上分配 x。
        check(cudaMalloc(&residual, bytes));  // 在 GPU 上分配 residual。
        check(cudaMalloc(&weight, hidden * sizeof(BF16)));  // 在 GPU 上分配一个长度为 hidden 的权重向量。
        check(cudaMalloc(&sum, bytes));  // 仅供拆分路径使用的 sum 缓冲区。
        check(cudaMalloc(&split, bytes));  // 分配拆分路径的输出。
        check(cudaMalloc(&fused, bytes));  // 分配融合路径的输出。
        check(cudaMemcpy(x, host_x.data(), bytes, cudaMemcpyHostToDevice));  // 将 x 从 CPU 拷贝到 GPU。
        check(cudaMemcpy(residual, host_r.data(), bytes, cudaMemcpyHostToDevice));  // 将 residual 从 CPU 拷贝到 GPU。
        check(cudaMemcpy(weight, host_w.data(), hidden * sizeof(BF16), cudaMemcpyHostToDevice));  // 将权重从 CPU 拷贝到 GPU。

        auto run_split = [&] {  // 定义一次拆分调用，按引用捕获设备指针和形状。
            add<<<(count + threads - 1) / threads, threads>>>(x, residual, sum, count);  // 先启动 Add，向上取整计算 block 数。这个 kernel 写入 sum。
            rmsnorm<<<batch, threads>>>(sum, weight, split);  // 同一默认 stream 再启动 RMSNorm，每行一个 block。该 kernel 读取 sum。
        };
        auto run_fused = [&] {  // 定义一次融合调用。
            add_rmsnorm<<<batch, threads>>>(x, residual, weight, fused);  // 每行一个 block，只提交一个 kernel。
        };
        for (int i = 0; i < (check_only ? 1 : 20); ++i) {  // 执行 20 轮预热。
            run_split();  // 提交一次拆分调用。
            run_fused();  // 提交一次融合调用。
        }
        check(cudaGetLastError());  // 检查 CUDA launch 错误。
        std::vector<BF16> a(count), b(count);  // 分配两份 CPU 端输出用于验证。
        check(cudaMemcpy(a.data(), split, bytes, cudaMemcpyDeviceToHost));  // 把拆分结果拷回 CPU。
        check(cudaMemcpy(b.data(), fused, bytes, cudaMemcpyDeviceToHost));  // 把融合结果拷回 CPU。
        for (int row = 0; row < batch; ++row) {  // CPU 参考按行计算。
            std::array<float, hidden> added{};  // 初始化长度为 hidden 的临时加法结果。
            double mean_square = 0;  // 用 double 累计 CPU 参考的均方。
            for (int col = 0; col < hidden; ++col) {  // 遍历当前行的列。
                int i = row * hidden + col;  // 将行列索引展开为数组偏移量。
                added[col] = __bfloat162float(__float2bfloat16(  // CPU 参考也对加法结果做一次 BF16 舍入，再转回 FP32。
                    __bfloat162float(host_x[i]) + __bfloat162float(host_r[i])));  // 把两个 BF16 输入转换到 FP32 后相加。
                mean_square += double(added[col]) * added[col] / hidden;  // 将当前平方除以 hidden 并累加。
            }
            for (int col = 0; col < hidden; ++col) {  // 再遍历每列，验证归一化后的结果。
                int i = row * hidden + col;  // 重新计算线性索引。
                float expected = added[col] * __bfloat162float(host_w[col]) /  // 构造参考输出的分子：加法结果乘上列权重。
                                 std::sqrt(mean_square + 1e-5);  // 除以整行的均方根，结果保存为 float。
                float actual = __bfloat162float(b[i]);  // 读取融合 BF16 输出并转 float。
                if (__bfloat162float(a[i]) != actual ||  // 检查拆分与融合的输出数值是否相等。
                    std::abs(actual - expected) > 1e-5f + 0.004f * std::abs(expected)) {  // 同时检查融合结果与 CPU 参考值的绝对误差是否超出容差。
                    std::fprintf(stderr, "Mismatch: batch=%d row=%d col=%d\n", batch, row, col);  // 报告首个失败的 batch、行、列。
                    return 2;  // 以退出码 2 中止，此 batch 不进入计时。
                }
            }
        }

        if (check_only) {  // 当前 batch 验算完成后直接退出。
            for (BF16* ptr : {x, residual, weight, sum, split, fused}) {
                check(cudaFree(ptr));
            }
            std::printf("PASS: batch=%d split and fused outputs match the CPU check\n", batch);
            return 0;
        }

        cudaEvent_t start, end;  // 声明测量区间的 CUDA events。
        check(cudaEventCreate(&start));  // 创建开始事件。
        check(cudaEventCreate(&end));  // 创建结束事件。
        auto measure = [&](auto run) {  // 定义可以接受任一运行路径的计时函数。
            check(cudaEventRecord(start));  // 在默认 stream 上记录开始事件。
            for (int i = 0; i < repeats; ++i) {  // 遍历本轮的 100 次调用。
                run();  // 提交当前路径的一次调用。
            }
            check(cudaGetLastError());  // 检查 launch 错误。
            check(cudaEventRecord(end));  // 在这些调用之后记录结束事件。
            check(cudaEventSynchronize(end));  // 等待结束事件完成，也就等到了该测量区间的设备任务执行完毕。
            float ms;  // 声明毫秒结果变量。
            check(cudaEventElapsedTime(&ms, start, end));  // 读取两个事件之间的时间差。
            return ms * 1000 / repeats;  // 转成微秒并除以调用次数，得到每次调用的平均延迟。
        };
        std::array<float, trials> split_times{}, fused_times{};  // 分别保存两条路径的 7 个测量值。
        for (int trial = 0; trial < trials; ++trial) {  // 逐轮进行配对测量。
            if (trial % 2 == 0) {  // 偶数轮先测拆分路径。
                split_times[trial] = measure(run_split);  // 保存拆分测量值。
                fused_times[trial] = measure(run_fused);  // 保存融合测量值。
            } else {  // 奇数轮先测融合，减轻固定测量顺序的影响。
                fused_times[trial] = measure(run_fused);  // 保存先测的融合延迟。
                split_times[trial] = measure(run_split);  // 保存后测的拆分延迟。
            }
        }
        std::sort(split_times.begin(), split_times.end());  // 排序拆分延迟，以便取中位数和范围。
        std::sort(fused_times.begin(), fused_times.end());  // 对融合延迟做同样的排序。
        float split_us = split_times[trials / 2];  // 取拆分延迟的中位数。
        float fused_us = fused_times[trials / 2];  // 取融合延迟的中位数。
        std::printf("%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", batch, split_us, fused_us,  // 输出 batch 和两条路径的中位延迟。
                    split_us / fused_us, split_times.front(), split_times.back(),  // 输出中位延迟之比，以及拆分路径的最小和最大延迟。
                    fused_times.front(), fused_times.back());  // 输出融合路径的最小和最大延迟。
        std::fflush(stdout);  // 刷新 stdout，使该 batch 的结果立即写入输出文件。
        check(cudaEventDestroy(start));  // 销毁开始事件。
        check(cudaEventDestroy(end));  // 销毁结束事件。
        for (BF16* ptr : {x, residual, weight, sum, split, fused}) {  // 遍历本 batch 分配的 GPU 缓冲区。
            check(cudaFree(ptr));  // 释放每个设备缓冲区。
        }
    }
}
