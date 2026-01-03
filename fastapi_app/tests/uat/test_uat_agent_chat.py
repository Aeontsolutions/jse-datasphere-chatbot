"""
Pytest integration for Agent Chat UAT tests.

Runs test cases defined in prompt_optimization_test_cases.json against
the /agent/chat endpoint and validates responses.

Usage:
    # Run all UAT tests against local server
    pytest tests/uat/test_uat_agent_chat.py -v -m uat

    # Run specific category
    pytest tests/uat/test_uat_agent_chat.py -v -k "tool_routing"

    # Run against deployed instance
    pytest tests/uat/ -v --uat-url https://api.example.com

    # Run with retry logic
    pytest tests/uat/ -v --uat-retries 3
"""

from typing import Any, Dict, List

import httpx
import pytest

from .uat_runner import TestResult, UATTestRunner


def _format_failure(result: TestResult) -> str:
    """Format failure message for assertion output."""
    lines = [f"Test {result.test_id} failed:"]
    if result.error:
        lines.append(f"  Error: {result.error}")
    if result.error_category.value != "none":
        lines.append(f"  Error Category: {result.error_category.value}")
    if result.retry_count > 0:
        lines.append(f"  Retries: {result.retry_count}")
    for v in result.validation_details:
        if not v.passed:
            lines.append(f"  {v.field}:")
            lines.append(f"    Expected: {v.expected}")
            lines.append(f"    Actual: {v.actual}")
            lines.append(f"    Message: {v.message}")
    return "\n".join(lines)


@pytest.mark.uat
@pytest.mark.tool_routing
@pytest.mark.asyncio
class TestToolRouting:
    """UAT tests for tool routing behavior."""

    @pytest.fixture
    def tool_routing_cases(self, test_cases: Dict[str, List[Dict]]) -> List[Dict]:
        """Get tool routing test cases."""
        return test_cases["tool_routing"]

    async def test_tool_routing_cases(
        self,
        uat_runner: UATTestRunner,
        tool_routing_cases: List[Dict],
        health_check,
    ):
        """Run all tool routing test cases."""
        async with httpx.AsyncClient() as client:
            for test_case in tool_routing_cases:
                result = await uat_runner.run_single_test(test_case, client)
                assert result.passed, _format_failure(result)


@pytest.mark.uat
@pytest.mark.prompt_optimization
@pytest.mark.asyncio
class TestPromptOptimization:
    """UAT tests for prompt optimization (pronoun resolution)."""

    @pytest.fixture
    def prompt_cases(self, test_cases: Dict[str, List[Dict]]) -> List[Dict]:
        """Get prompt optimization test cases."""
        return test_cases["prompt_optimization"]

    async def test_prompt_optimization_cases(
        self,
        uat_runner: UATTestRunner,
        prompt_cases: List[Dict],
        health_check,
    ):
        """Run all prompt optimization test cases."""
        async with httpx.AsyncClient() as client:
            for test_case in prompt_cases:
                result = await uat_runner.run_single_test(test_case, client)
                assert result.passed, _format_failure(result)


@pytest.mark.uat
@pytest.mark.clarification
@pytest.mark.asyncio
class TestClarification:
    """UAT tests for clarification flow."""

    @pytest.fixture
    def clarification_cases(self, test_cases: Dict[str, List[Dict]]) -> List[Dict]:
        """Get clarification test cases."""
        return test_cases["clarification"]

    async def test_clarification_cases(
        self,
        uat_runner: UATTestRunner,
        clarification_cases: List[Dict],
        health_check,
    ):
        """Run all clarification test cases."""
        async with httpx.AsyncClient() as client:
            for test_case in clarification_cases:
                result = await uat_runner.run_single_test(test_case, client)
                assert result.passed, _format_failure(result)


@pytest.mark.uat
@pytest.mark.defaults
@pytest.mark.asyncio
class TestDefaults:
    """UAT tests for default value application."""

    @pytest.fixture
    def defaults_cases(self, test_cases: Dict[str, List[Dict]]) -> List[Dict]:
        """Get defaults test cases."""
        return test_cases["defaults"]

    async def test_defaults_cases(
        self,
        uat_runner: UATTestRunner,
        defaults_cases: List[Dict],
        health_check,
    ):
        """Run all defaults test cases."""
        async with httpx.AsyncClient() as client:
            for test_case in defaults_cases:
                result = await uat_runner.run_single_test(test_case, client)
                assert result.passed, _format_failure(result)


@pytest.mark.uat
@pytest.mark.edge_cases
@pytest.mark.asyncio
class TestEdgeCases:
    """UAT tests for edge cases."""

    @pytest.fixture
    def edge_cases_list(self, test_cases: Dict[str, List[Dict]]) -> List[Dict]:
        """Get edge case test cases."""
        return test_cases["edge_cases"]

    async def test_edge_cases(
        self,
        uat_runner: UATTestRunner,
        edge_cases_list: List[Dict],
        health_check,
    ):
        """Run all edge case tests."""
        async with httpx.AsyncClient() as client:
            for test_case in edge_cases_list:
                result = await uat_runner.run_single_test(test_case, client)

                # Edge cases may intentionally fail gracefully
                expected_behavior = test_case.get("expected_behavior", "")
                if expected_behavior == "proceed_gracefully":
                    # Just verify we got some response
                    assert (
                        result.actual.get("response_length", 0) > 0 or result.error is None
                    ), f"Test {test_case['id']} should proceed gracefully but got error: {result.error}"
                else:
                    assert result.passed, _format_failure(result)


# Individual parametrized tests for more granular CI reporting
@pytest.mark.uat
@pytest.mark.asyncio
class TestIndividualCases:
    """Individual test cases for detailed reporting in CI."""

    async def _run_test_by_id(
        self,
        test_id: str,
        uat_runner: UATTestRunner,
        test_cases: Dict[str, List[Dict]],
    ) -> TestResult:
        """Run a specific test case by ID."""
        # Find the test case
        for category_cases in test_cases.values():
            for tc in category_cases:
                if tc["id"] == test_id:
                    async with httpx.AsyncClient() as client:
                        return await uat_runner.run_single_test(tc, client)
        pytest.fail(f"Test case not found: {test_id}")

    @pytest.mark.parametrize(
        "test_id",
        [
            "route_financial_only_revenue",
            "route_financial_only_eps",
            "route_web_only_news",
            "route_both_vague_performance_symbol",
        ],
    )
    async def test_routing_individual(
        self,
        test_id: str,
        uat_runner: UATTestRunner,
        test_cases: Dict[str, List[Dict]],
        health_check,
    ):
        """Test individual routing cases for detailed CI reporting."""
        result = await self._run_test_by_id(test_id, uat_runner, test_cases)
        assert result.passed, _format_failure(result)

    @pytest.mark.parametrize(
        "test_id",
        [
            "clarify_no_entity",
            "clarify_unresolved_pronoun",
            "no_clarify_general_market",
        ],
    )
    async def test_clarification_individual(
        self,
        test_id: str,
        uat_runner: UATTestRunner,
        test_cases: Dict[str, List[Dict]],
        health_check,
    ):
        """Test individual clarification cases for detailed CI reporting."""
        result = await self._run_test_by_id(test_id, uat_runner, test_cases)
        assert result.passed, _format_failure(result)
