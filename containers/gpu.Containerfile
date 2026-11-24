FROM docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        ninja-build \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/kernel-agent-venv

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/opt/kernel-agent-venv/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
        'torch==2.13.0+cu130' \
    && python -m pip install --no-cache-dir 'flashinfer-python==0.6.4' setuptools \
    && python -m pip check \
    && nvcc --version \
    && compute-sanitizer --version \
    && ncu --version \
    && mkdir -p /opt/kernel-agent/harness /opt/kernel-agent/tasks

WORKDIR /opt/kernel-agent
USER 65534:65534
ENTRYPOINT []
