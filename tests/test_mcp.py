"""Regression coverage for the structured MCP response and run scoping."""

import asyncio
import unittest
from tempfile import TemporaryDirectory

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from harness.__main__ import _local_mcp
from harness.evaluator import Evaluator
from harness.results import RunStore
from tasks.rmsnorm import PUBLIC_CASES


class MCPTest(unittest.TestCase):
    def test_mock_feedback_is_structured_and_run_scoped(self):
        with TemporaryDirectory() as directory:
            store = RunStore(directory)
            config = {"variant": "B", "backend": "mock", "evaluation_budget": 2,
                      "public_cases": PUBLIC_CASES, "warmups": 1, "samples": 3}
            run_id, token = store.create(config)
            other_id, _ = store.create(config)
            with _local_mcp(Evaluator(store), run_id) as endpoint:
                asyncio.run(self._check_session(endpoint, run_id, other_id, token))

    async def _check_session(self, endpoint, run_id, other_id, token):
        async with streamablehttp_client(endpoint, headers={"Authorization": f"Bearer {token}"}) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools = await session.list_tools()
                self.assertEqual([tool.name for tool in tools.tools], ["evaluate_kernel"])

                async def submit(run, source):
                    return await session.call_tool("evaluate_kernel", {"run_id": run, "source": source})

                self.assertTrue((await submit(other_id, "void run() {}")).isError)
                failed = await submit(run_id, "void run() {} // mock: compile_error")
                self.assertEqual(failed.structuredContent["status"], "compile_error")
                passed = await submit(run_id, "void run() {}")
                feedback = passed.structuredContent
                self.assertEqual(feedback["status"], "passed")
                self.assertNotIn("best_candidate_id", feedback)
                self.assertTrue(all("seed" not in case for case in feedback["cases"]))
                self.assertTrue((await submit(run_id, "void run() {}")).isError)
