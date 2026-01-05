#!/usr/bin/env python3
"""
Test runner for streaming functionality tests.
This script provides an easy way to run streaming tests with different options.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, parallel=False):
    """Run the streaming tests with the specified options."""

    # Change to the fastapi_app directory
    fastapi_dir = Path(__file__).parent.parent
    os.chdir(fastapi_dir)

    # Build the pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test files based on type
    if test_type == "unit":
        cmd.extend(["tests/test_streaming_units.py"])
    elif test_type == "integration":
        cmd.extend(["tests/test_streaming.py"])
    elif test_type == "all":
        cmd.extend(["tests/test_streaming_units.py", "tests/test_streaming.py"])
    else:
        print(f"Unknown test type: {test_type}")
        return False

    # Add options
    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])

    if parallel:
        cmd.extend(["-n", "auto"])

    # Add common options
    cmd.extend(["--tb=short", "--strict-markers", "--disable-warnings"])

    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
        print("=" * 60)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"❌ Tests failed with exit code: {e.returncode}")
        return False


def run_specific_test(test_name, verbose=False):
    """Run a specific test by name."""
    fastapi_dir = Path(__file__).parent.parent
    os.chdir(fastapi_dir)

    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    cmd.extend(["-k", test_name, "tests/test_streaming_units.py", "--tb=short"])

    print(f"Running specific test: {test_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
        print("=" * 60)
        print("✅ Test passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"❌ Test failed with exit code: {e.returncode}")
        return False


def list_tests():
    """List all available tests."""
    fastapi_dir = Path(__file__).parent.parent
    os.chdir(fastapi_dir)

    cmd = ["python", "-m", "pytest", "--collect-only", "-q", "tests/test_streaming_units.py"]

    print("Available streaming tests:")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.split("\n")

        for line in lines:
            if "::" in line and "test_" in line:
                test_name = line.strip()
                if test_name:
                    print(f"  {test_name}")

        print("=" * 60)
        print("To run a specific test, use: python run_streaming_tests.py --test-name <test_name>")

    except subprocess.CalledProcessError as e:
        print(f"Error listing tests: {e}")


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Run streaming functionality tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_streaming_tests.py                    # Run all tests
  python run_streaming_tests.py --type unit        # Run only unit tests
  python run_streaming_tests.py --type integration # Run only integration tests
  python run_streaming_tests.py --verbose          # Run with verbose output
  python run_streaming_tests.py --coverage         # Run with coverage report
  python run_streaming_tests.py --list             # List all available tests
  python run_streaming_tests.py --test-name test_emit_progress  # Run specific test
        """,
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["all", "unit", "integration"],
        default="all",
        help="Type of tests to run (default: all)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests with verbose output"
    )

    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")

    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)",
    )

    parser.add_argument("--list", "-l", action="store_true", help="List all available tests")

    parser.add_argument("--test-name", "-n", help="Run a specific test by name")

    args = parser.parse_args()

    if args.list:
        list_tests()
        return

    if args.test_name:
        success = run_specific_test(args.test_name, args.verbose)
    else:
        success = run_tests(args.type, args.verbose, args.coverage, args.parallel)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
