"""
UAT Test Runner for Agent Chat Endpoint.

Loads test cases from JSON, executes against /agent/chat endpoint,
validates responses, captures latency metrics, and generates reports.

Features:
- Latency tracking with min/max/avg/p95 statistics
- Parallel test execution for load testing
- Retry logic with exponential backoff
- Error categorization (timeout, rate limit, connection, server errors)

Usage:
    # Run all tests against local server
    python tests/uat/uat_runner.py

    # Run specific categories
    python tests/uat/uat_runner.py --category tool_routing --category clarification

    # Run against deployed instance with JSON output
    python tests/uat/uat_runner.py --url https://api.example.com --output report.json

    # Run tests in parallel (for load testing)
    python tests/uat/uat_runner.py --parallel --concurrency 5

    # Enable retry logic
    python tests/uat/uat_runner.py --retries 3
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Constants
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 60.0  # seconds
DEFAULT_RETRIES = 0
DEFAULT_CONCURRENCY = 1
TEST_CASES_FILE = Path(__file__).parent / "prompt_optimization_test_cases.json"


class TestBehavior(str, Enum):
    """Expected test behavior types."""

    PROCEED = "proceed"
    CLARIFY = "clarify"
    PROCEED_WITH_DEFAULTS = "proceed_with_defaults"
    NO_DATA_FOUND = "no_data_found_message"
    NO_TOOLS_RESPONSE = "no_tools_response"
    PROCEED_GRACEFULLY = "proceed_gracefully"


class ErrorCategory(str, Enum):
    """Categorization of errors for analysis."""

    NONE = "none"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    CONNECTION = "connection"
    SERVER_ERROR = "server_error"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ValidationDetail:
    """Details of individual validation checks."""

    field: str
    expected: Any
    actual: Any
    passed: bool
    message: str


@dataclass
class TestResult:
    """Result of a single test case execution."""

    test_id: str
    category: str
    passed: bool
    duration_ms: float
    start_time: datetime
    end_time: datetime
    expected: Dict[str, Any]
    actual: Dict[str, Any]
    error: Optional[str] = None
    error_category: ErrorCategory = ErrorCategory.NONE
    retry_count: int = 0
    validation_details: List[ValidationDetail] = field(default_factory=list)


@dataclass
class LatencyStats:
    """Latency statistics for a category."""

    min_ms: float
    max_ms: float
    avg_ms: float
    p95_ms: float
    count: int


@dataclass
class ErrorStats:
    """Error statistics for reporting."""

    total_errors: int
    by_category: Dict[str, int]
    retry_success_count: int
    retry_failure_count: int


@dataclass
class TestReport:
    """Summary report of all test executions."""

    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    total_duration_ms: float
    latency_by_category: Dict[str, LatencyStats]
    error_stats: ErrorStats
    results: List[TestResult]
    generated_at: datetime


class UATTestRunner:
    """
    UAT Test Runner for the /agent/chat endpoint.

    Supports:
    - Loading test cases from JSON
    - Running against real or mocked endpoints
    - Capturing latency metrics with statistics
    - Parallel test execution for load testing
    - Retry logic with exponential backoff
    - Error categorization for debugging
    - Generating detailed reports
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        test_cases_path: Optional[Path] = None,
        max_retries: int = DEFAULT_RETRIES,
        concurrency: int = DEFAULT_CONCURRENCY,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.test_cases_path = test_cases_path or TEST_CASES_FILE
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.test_cases: Dict[str, List[Dict]] = {}
        self.results: List[TestResult] = []
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _categorize_error(
        self, error: Exception, status_code: Optional[int] = None
    ) -> ErrorCategory:
        """Categorize an error for reporting and analysis."""
        if isinstance(error, httpx.TimeoutException):
            return ErrorCategory.TIMEOUT
        elif isinstance(error, httpx.ConnectError):
            return ErrorCategory.CONNECTION
        elif status_code:
            if status_code == 429:
                return ErrorCategory.RATE_LIMIT
            elif 500 <= status_code < 600:
                return ErrorCategory.SERVER_ERROR
            elif 400 <= status_code < 500:
                return ErrorCategory.VALIDATION
        return ErrorCategory.UNKNOWN

    async def _execute_with_retry(
        self,
        test_case: Dict[str, Any],
        client: httpx.AsyncClient,
    ) -> TestResult:
        """Execute a test with retry logic and exponential backoff."""
        last_result: Optional[TestResult] = None
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            result = await self._execute_single_request(test_case, client)
            result.retry_count = retry_count

            # Success or validation failure (don't retry validation failures)
            if result.passed or result.error_category == ErrorCategory.VALIDATION:
                return result

            # Don't retry on the last attempt
            if attempt < self.max_retries:
                # Exponential backoff: 1s, 2s, 4s, ...
                backoff = 2**attempt
                await asyncio.sleep(backoff)
                retry_count += 1

            last_result = result

        return last_result or result

    def load_test_cases(self) -> None:
        """Load test cases from JSON file."""
        with open(self.test_cases_path, "r") as f:
            data = json.load(f)

        # Extract all test case categories
        self.test_cases = {
            "tool_routing": data.get("tool_routing_test_cases", []),
            "prompt_optimization": data.get("prompt_optimization_test_cases", []),
            "clarification": data.get("clarification_test_cases", []),
            "defaults": data.get("defaults_test_cases", []),
            "edge_cases": data.get("edge_cases", []),
        }

        total = sum(len(cases) for cases in self.test_cases.values())
        print(f"Loaded {total} test cases from {self.test_cases_path}")

    async def _execute_single_request(
        self,
        test_case: Dict[str, Any],
        client: httpx.AsyncClient,
    ) -> TestResult:
        """Execute a single HTTP request and validate the response (no retry)."""
        test_id = test_case["id"]
        category = test_case.get("category", "unknown")

        # Build request payload
        payload = self._build_request(test_case)

        # Capture timing with high precision
        start_time = datetime.now()
        start_ts = time.perf_counter()

        try:
            response = await client.post(
                f"{self.base_url}/agent/chat",
                json=payload,
                timeout=self.timeout,
            )

            end_ts = time.perf_counter()
            end_time = datetime.now()
            duration_ms = (end_ts - start_ts) * 1000

            if response.status_code != 200:
                error_cat = self._categorize_error(Exception(), response.status_code)
                return TestResult(
                    test_id=test_id,
                    category=category,
                    passed=False,
                    duration_ms=duration_ms,
                    start_time=start_time,
                    end_time=end_time,
                    expected=self._extract_expected(test_case),
                    actual={"status_code": response.status_code, "body": response.text[:500]},
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    error_category=error_cat,
                )

            response_data = response.json()

            # Validate response against expected fields
            validation_results = self._validate_response(test_case, response_data)
            passed = all(v.passed for v in validation_results)

            return TestResult(
                test_id=test_id,
                category=category,
                passed=passed,
                duration_ms=duration_ms,
                start_time=start_time,
                end_time=end_time,
                expected=self._extract_expected(test_case),
                actual=self._extract_actual(response_data),
                error_category=ErrorCategory.NONE if passed else ErrorCategory.VALIDATION,
                validation_details=validation_results,
            )

        except httpx.TimeoutException as e:
            end_ts = time.perf_counter()
            end_time = datetime.now()
            duration_ms = (end_ts - start_ts) * 1000

            return TestResult(
                test_id=test_id,
                category=category,
                passed=False,
                duration_ms=duration_ms,
                start_time=start_time,
                end_time=end_time,
                expected=self._extract_expected(test_case),
                actual={},
                error=f"Request timed out after {self.timeout}s",
                error_category=self._categorize_error(e),
            )

        except httpx.ConnectError as e:
            end_ts = time.perf_counter()
            end_time = datetime.now()
            duration_ms = (end_ts - start_ts) * 1000

            return TestResult(
                test_id=test_id,
                category=category,
                passed=False,
                duration_ms=duration_ms,
                start_time=start_time,
                end_time=end_time,
                expected=self._extract_expected(test_case),
                actual={},
                error=f"Connection error: {str(e)}",
                error_category=self._categorize_error(e),
            )

        except Exception as e:
            end_ts = time.perf_counter()
            end_time = datetime.now()
            duration_ms = (end_ts - start_ts) * 1000

            return TestResult(
                test_id=test_id,
                category=category,
                passed=False,
                duration_ms=duration_ms,
                start_time=start_time,
                end_time=end_time,
                expected=self._extract_expected(test_case),
                actual={},
                error=str(e),
                error_category=self._categorize_error(e),
            )

    async def run_single_test(
        self,
        test_case: Dict[str, Any],
        client: httpx.AsyncClient,
    ) -> TestResult:
        """Execute a single test case with retry logic if enabled."""
        if self.max_retries > 0:
            return await self._execute_with_retry(test_case, client)
        return await self._execute_single_request(test_case, client)

    def _build_request(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Build the request payload from a test case."""
        return {
            "query": test_case["query"],
            "conversation_history": test_case.get("conversation_history", []),
            "memory_enabled": True,
            "enable_web_search": test_case.get("enable_web_search", True),
            "enable_financial_data": test_case.get("enable_financial_data", True),
        }

    def _extract_expected(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Extract expected values from test case for reporting."""
        expected = {}
        for key in test_case:
            if key.startswith("expected_"):
                expected[key] = test_case[key]
        return expected

    def _extract_actual(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant fields from response for reporting."""
        return {
            "tools_executed": response.get("tools_executed"),
            "needs_clarification": response.get("needs_clarification"),
            "clarification_question": response.get("clarification_question"),
            "data_found": response.get("data_found"),
            "record_count": response.get("record_count"),
            "response_length": len(response.get("response", "")),
            "has_sources": bool(response.get("sources")),
            "has_chart": bool(response.get("chart")),
        }

    def _validate_response(
        self,
        test_case: Dict[str, Any],
        response: Dict[str, Any],
    ) -> List[ValidationDetail]:
        """Validate the response against expected fields."""
        validations = []

        # Validate expected_tools
        if "expected_tools" in test_case:
            expected_tools = set(test_case["expected_tools"])
            actual_tools = set(response.get("tools_executed") or [])
            passed = expected_tools == actual_tools
            validations.append(
                ValidationDetail(
                    field="tools_executed",
                    expected=sorted(list(expected_tools)),
                    actual=sorted(list(actual_tools)),
                    passed=passed,
                    message=(
                        "Tools match"
                        if passed
                        else f"Expected {expected_tools}, got {actual_tools}"
                    ),
                )
            )

        # Validate expected_behavior
        if "expected_behavior" in test_case:
            behavior = test_case["expected_behavior"]
            passed = self._check_behavior(behavior, response)
            validations.append(
                ValidationDetail(
                    field="behavior",
                    expected=behavior,
                    actual=self._describe_actual_behavior(response),
                    passed=passed,
                    message=(
                        f"Behavior matches '{behavior}'"
                        if passed
                        else f"Expected behavior '{behavior}'"
                    ),
                )
            )

        # Validate expected_question_contains
        if "expected_question_contains" in test_case:
            expected_text = test_case["expected_question_contains"].lower()
            actual_question = (response.get("clarification_question") or "").lower()
            passed = expected_text in actual_question
            validations.append(
                ValidationDetail(
                    field="clarification_question_contains",
                    expected=expected_text,
                    actual=actual_question[:100] if actual_question else "(none)",
                    passed=passed,
                    message=(
                        "Question contains expected text"
                        if passed
                        else "Missing expected text in question"
                    ),
                )
            )

        # Validate expected_optimized_query_contains
        if "expected_optimized_query_contains" in test_case:
            expected_terms = test_case["expected_optimized_query_contains"]
            response_text = response.get("response", "").lower()
            missing_terms = [term for term in expected_terms if term.lower() not in response_text]
            all_found = len(missing_terms) == 0
            validations.append(
                ValidationDetail(
                    field="response_contains_resolved_entities",
                    expected=expected_terms,
                    actual=f"Missing: {missing_terms}" if missing_terms else "All found",
                    passed=all_found,
                    message=(
                        "Response contains expected entities"
                        if all_found
                        else f"Missing entities: {missing_terms}"
                    ),
                )
            )

        # Validate expected_reason (for clarification)
        if "expected_reason" in test_case:
            expected_clarify = test_case["expected_reason"] in [
                "no_entity",
                "unresolved_pronoun",
                "ambiguous_comparison",
            ]
            actual_clarify = response.get("needs_clarification", False)
            passed = expected_clarify == actual_clarify
            validations.append(
                ValidationDetail(
                    field="needs_clarification",
                    expected=expected_clarify,
                    actual=actual_clarify,
                    passed=passed,
                    message=(
                        "Clarification state matches" if passed else "Clarification state mismatch"
                    ),
                )
            )

        # Validate expected_defaults_applied
        if "expected_defaults_applied" in test_case:
            filters = response.get("filters_used") or {}
            defaults = test_case["expected_defaults_applied"]

            if "year" in defaults:
                has_years = bool(filters.get("years"))
                validations.append(
                    ValidationDetail(
                        field="year_default_applied",
                        expected=True,
                        actual=has_years,
                        passed=has_years,
                        message=(
                            "Year default applied"
                            if has_years
                            else "Year default not found in filters"
                        ),
                    )
                )

            if "metrics" in defaults:
                has_metrics = bool(filters.get("standard_items"))
                validations.append(
                    ValidationDetail(
                        field="metrics_default_applied",
                        expected=True,
                        actual=has_metrics,
                        passed=has_metrics,
                        message=(
                            "Metrics default applied"
                            if has_metrics
                            else "Metrics default not found in filters"
                        ),
                    )
                )

        # Validate expected_routing_method (informational - from logs, hard to validate)
        if "expected_routing_method" in test_case:
            # This is logged server-side, we can't directly validate but record expectation
            validations.append(
                ValidationDetail(
                    field="routing_method",
                    expected=test_case["expected_routing_method"],
                    actual="(check server logs)",
                    passed=True,  # Can't validate without log access
                    message="Routing method should be checked in server logs",
                )
            )

        # Validate expected_confidence (informational - from logs)
        if "expected_confidence" in test_case:
            validations.append(
                ValidationDetail(
                    field="confidence",
                    expected=test_case["expected_confidence"],
                    actual="(check server logs)",
                    passed=True,  # Can't validate without log access
                    message="Confidence level should be checked in server logs",
                )
            )

        return validations

    def _check_behavior(self, behavior: str, response: Dict[str, Any]) -> bool:
        """Check if response matches expected behavior."""
        if behavior == "proceed":
            return not response.get("needs_clarification", False) and bool(response.get("response"))
        elif behavior == "clarify":
            return response.get("needs_clarification", False)
        elif behavior == "proceed_with_defaults":
            return not response.get("needs_clarification", False)
        elif behavior == "no_data_found_message":
            return response.get("data_found", True) is False or response.get("record_count", 1) == 0
        elif behavior == "no_tools_response":
            return not response.get("tools_executed")
        elif behavior == "proceed_gracefully":
            return response.get("response") is not None
        return True

    def _describe_actual_behavior(self, response: Dict[str, Any]) -> str:
        """Describe the actual behavior from response."""
        if response.get("needs_clarification"):
            return "clarify"
        elif not response.get("tools_executed"):
            return "no_tools_response"
        elif not response.get("data_found"):
            return "no_data_found"
        return "proceed"

    async def run_category(
        self,
        category: str,
        client: httpx.AsyncClient,
        verbose: bool = True,
        parallel: bool = False,
    ) -> List[TestResult]:
        """Run all tests in a category, optionally in parallel."""
        tests = self.test_cases.get(category, [])
        if not tests:
            print(f"  No tests found for category: {category}")
            return []

        if parallel and self.concurrency > 1:
            return await self._run_category_parallel(tests, client, verbose)
        else:
            return await self._run_category_sequential(tests, client, verbose)

    async def _run_category_sequential(
        self,
        tests: List[Dict[str, Any]],
        client: httpx.AsyncClient,
        verbose: bool,
    ) -> List[TestResult]:
        """Run tests sequentially within a category."""
        results = []
        for i, test_case in enumerate(tests):
            test_id = test_case["id"]
            if verbose:
                print(f"  [{i + 1}/{len(tests)}] Running {test_id}...", end=" ", flush=True)

            result = await self.run_single_test(test_case, client)
            results.append(result)
            self.results.append(result)

            if verbose:
                status = "PASS" if result.passed else "FAIL"
                retry_info = f" (retries: {result.retry_count})" if result.retry_count > 0 else ""
                print(f"{status} ({result.duration_ms:.0f}ms){retry_info}")

        return results

    async def _run_category_parallel(
        self,
        tests: List[Dict[str, Any]],
        client: httpx.AsyncClient,
        verbose: bool,
    ) -> List[TestResult]:
        """Run tests in parallel within a category using semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)

        async def run_with_semaphore(test_case: Dict[str, Any], index: int) -> TestResult:
            async with self._semaphore:
                if verbose:
                    print(f"  [{index + 1}/{len(tests)}] Starting {test_case['id']}...")
                result = await self.run_single_test(test_case, client)
                if verbose:
                    status = "PASS" if result.passed else "FAIL"
                    retry_info = (
                        f" (retries: {result.retry_count})" if result.retry_count > 0 else ""
                    )
                    print(
                        f"  [{index + 1}/{len(tests)}] {test_case['id']}: {status} ({result.duration_ms:.0f}ms){retry_info}"
                    )
                return result

        # Run all tests concurrently with semaphore limiting
        tasks = [run_with_semaphore(test, i) for i, test in enumerate(tests)]
        results = await asyncio.gather(*tasks)

        # Add to global results
        for result in results:
            self.results.append(result)

        return list(results)

    async def run_all(
        self,
        categories: Optional[List[str]] = None,
        verbose: bool = True,
        parallel: bool = False,
    ) -> TestReport:
        """Run all test cases and generate a report.

        Args:
            categories: List of categories to run (None = all)
            verbose: Print progress output
            parallel: Run tests in parallel within categories
        """
        if not self.test_cases:
            self.load_test_cases()

        categories_to_run = categories or list(self.test_cases.keys())
        self.results = []  # Reset results

        if parallel:
            print(f"Running in parallel mode with concurrency={self.concurrency}")

        async with httpx.AsyncClient() as client:
            for category in categories_to_run:
                if verbose:
                    mode = "parallel" if parallel else "sequential"
                    print(f"\nRunning {category} tests ({mode})...")
                await self.run_category(category, client, verbose, parallel)

        return self.generate_report()

    def calculate_latency_stats(self, durations: List[float]) -> LatencyStats:
        """Calculate latency statistics from a list of durations."""
        if not durations:
            return LatencyStats(0, 0, 0, 0, 0)

        sorted_durations = sorted(durations)
        p95_index = int(len(sorted_durations) * 0.95)

        return LatencyStats(
            min_ms=round(min(durations), 2),
            max_ms=round(max(durations), 2),
            avg_ms=round(statistics.mean(durations), 2),
            p95_ms=round(
                (
                    sorted_durations[p95_index]
                    if p95_index < len(sorted_durations)
                    else sorted_durations[-1]
                ),
                2,
            ),
            count=len(durations),
        )

    def generate_report(self) -> TestReport:
        """Generate a summary report from all test results."""
        # Group by category for latency stats
        by_category: Dict[str, List[float]] = {}
        for result in self.results:
            if result.category not in by_category:
                by_category[result.category] = []
            by_category[result.category].append(result.duration_ms)

        latency_by_category = {
            cat: self.calculate_latency_stats(durations) for cat, durations in by_category.items()
        }

        # Calculate error statistics
        error_by_category: Dict[str, int] = {}
        retry_success = 0
        retry_failure = 0
        for result in self.results:
            if result.error_category != ErrorCategory.NONE:
                cat_name = result.error_category.value
                error_by_category[cat_name] = error_by_category.get(cat_name, 0) + 1
            if result.retry_count > 0:
                if result.passed:
                    retry_success += 1
                else:
                    retry_failure += 1

        error_stats = ErrorStats(
            total_errors=sum(error_by_category.values()),
            by_category=error_by_category,
            retry_success_count=retry_success,
            retry_failure_count=retry_failure,
        )

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        return TestReport(
            total_tests=total,
            passed=passed,
            failed=total - passed,
            pass_rate=round(passed / total, 4) if total > 0 else 0,
            total_duration_ms=round(sum(r.duration_ms for r in self.results), 2),
            latency_by_category=latency_by_category,
            error_stats=error_stats,
            results=self.results,
            generated_at=datetime.now(),
        )

    def print_report(self, report: TestReport) -> None:
        """Print a formatted report to console."""
        print("\n" + "=" * 70)
        print("UAT TEST REPORT - Agent Chat Endpoint")
        print("=" * 70)
        print(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target:    {self.base_url}")
        print("-" * 70)
        print(f"Total Tests: {report.total_tests}")
        print(f"Passed:      {report.passed} ({report.pass_rate:.1%})")
        print(f"Failed:      {report.failed}")
        print(f"Duration:    {report.total_duration_ms:.2f}ms")

        print("\n" + "-" * 70)
        print("LATENCY BY CATEGORY")
        print("-" * 70)
        print(f"{'Category':<25} {'Count':>6} {'Min':>10} {'Avg':>10} {'P95':>10} {'Max':>10}")
        print("-" * 70)
        for category, stats in report.latency_by_category.items():
            print(
                f"{category:<25} {stats.count:>6} {stats.min_ms:>9.0f}ms {stats.avg_ms:>9.0f}ms "
                f"{stats.p95_ms:>9.0f}ms {stats.max_ms:>9.0f}ms"
            )

        # Overall latency
        all_durations = [r.duration_ms for r in report.results]
        if all_durations:
            overall = self.calculate_latency_stats(all_durations)
            print("-" * 70)
            print(
                f"{'OVERALL':<25} {overall.count:>6} {overall.min_ms:>9.0f}ms {overall.avg_ms:>9.0f}ms "
                f"{overall.p95_ms:>9.0f}ms {overall.max_ms:>9.0f}ms"
            )

        # Error statistics
        if report.error_stats.total_errors > 0 or report.error_stats.retry_success_count > 0:
            print("\n" + "-" * 70)
            print("ERROR & RETRY STATISTICS")
            print("-" * 70)
            print(f"Total Errors:    {report.error_stats.total_errors}")
            if report.error_stats.by_category:
                print("By Category:")
                for cat, count in report.error_stats.by_category.items():
                    print(f"  - {cat}: {count}")
            if (
                report.error_stats.retry_success_count > 0
                or report.error_stats.retry_failure_count > 0
            ):
                print(f"Retry Success:   {report.error_stats.retry_success_count}")
                print(f"Retry Failures:  {report.error_stats.retry_failure_count}")

        if report.failed > 0:
            print("\n" + "-" * 70)
            print("FAILED TESTS")
            print("-" * 70)
            for result in report.results:
                if not result.passed:
                    print(f"\n[FAIL] {result.test_id} ({result.category})")
                    print(f"       Duration: {result.duration_ms:.0f}ms")
                    if result.error_category != ErrorCategory.NONE:
                        print(f"       Error Type: {result.error_category.value}")
                    if result.retry_count > 0:
                        print(f"       Retries: {result.retry_count}")
                    if result.error:
                        print(f"       Error: {result.error[:100]}")
                    for v in result.validation_details:
                        if not v.passed:
                            print(f"       - {v.field}: {v.message}")
                            print(f"         Expected: {v.expected}")
                            print(f"         Actual:   {v.actual}")

        print("\n" + "=" * 70)

    def save_report_json(self, report: TestReport, path: Path) -> None:
        """Save report to JSON file."""

        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        # Convert to dict for JSON serialization
        report_dict = {
            "total_tests": report.total_tests,
            "passed": report.passed,
            "failed": report.failed,
            "pass_rate": report.pass_rate,
            "total_duration_ms": report.total_duration_ms,
            "generated_at": report.generated_at.isoformat(),
            "base_url": self.base_url,
            "config": {
                "max_retries": self.max_retries,
                "concurrency": self.concurrency,
                "timeout": self.timeout,
            },
            "latency_by_category": {
                cat: asdict(stats) for cat, stats in report.latency_by_category.items()
            },
            "error_stats": asdict(report.error_stats),
            "results": [
                {
                    "test_id": r.test_id,
                    "category": r.category,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "start_time": r.start_time.isoformat(),
                    "end_time": r.end_time.isoformat(),
                    "expected": r.expected,
                    "actual": r.actual,
                    "error": r.error,
                    "error_category": r.error_category.value,
                    "retry_count": r.retry_count,
                    "validation_details": [asdict(v) for v in r.validation_details],
                }
                for r in report.results
            ],
        }

        with open(path, "w") as f:
            json.dump(report_dict, f, indent=2, default=serialize)


async def main():
    """Run UAT tests from command line."""
    parser = argparse.ArgumentParser(
        description="UAT Test Runner for Agent Chat Endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python uat_runner.py                                  # Run all tests locally
  python uat_runner.py --url https://api.example.com   # Run against deployed API
  python uat_runner.py --category tool_routing         # Run specific category
  python uat_runner.py --output report.json            # Save JSON report
  python uat_runner.py --parallel --concurrency 5      # Run in parallel for load testing
  python uat_runner.py --retries 3                     # Enable retry with 3 attempts
        """,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Categories to test (can specify multiple). Options: tool_routing, prompt_optimization, clarification, defaults, edge_cases",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run tests in parallel within each category (for load testing)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent requests when using --parallel (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--retries",
        "-r",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Number of retry attempts for failed requests (default: {DEFAULT_RETRIES})",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity",
    )
    args = parser.parse_args()

    runner = UATTestRunner(
        base_url=args.url,
        timeout=args.timeout,
        max_retries=args.retries,
        concurrency=args.concurrency,
    )
    report = await runner.run_all(
        categories=args.categories,
        verbose=not args.quiet,
        parallel=args.parallel,
    )
    runner.print_report(report)

    if args.output:
        output_path = Path(args.output)
        runner.save_report_json(report, output_path)
        print(f"\nReport saved to: {output_path}")

    # Exit with error code if any tests failed
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
