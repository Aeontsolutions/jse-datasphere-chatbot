# Archived Documentation

This directory contains historical documentation from previous implementation phases.

These documents are kept for reference but may not reflect the current architecture:

- **ASYNC_S3_IMPLEMENTATION_SUMMARY.md** - Original async S3 download implementation
- **POLLING_IMPLEMENTATION_GUIDE.md** - Polling mechanism for async jobs
- **PRODUCTION_POLLING_SETUP.md** - Production deployment of polling
- **QUICK_FIX_404_POLLING_ERROR.md** - Historical bug fix documentation
- **REDIS_IMPLEMENTATION_GUIDE.md** - Redis async job storage implementation
  (the job/Redis polling code these three describe was itself archived on
  2026-08-14 - see `fastapi_app/app/_archive/README.md`)
- **task.md** - Original task tracking
- **MIGRATION_SUMMARY.md** - `requirements.txt` → `pyproject.toml` migration (2025-12-17)
- **PHASE1_COMPLETION_SUMMARY.md** - Phase 1 of the maintainability refactor (Dec 2025)
- **REFACTOR_PLAN.md** - The original 3-phase maintainability refactoring plan (Dec 2025); superseded by the 2026-08-14 prod-cleanup pass
- **LEGACY_TEST_FIXES.md**, **TODO_TEST_FIXES.md**, **TEST_INFRASTRUCTURE_SUMMARY.md** - Point-in-time test-suite status reports from the same refactor

## Current Documentation

For current system documentation, see:
- [DEVELOPMENT.md](../DEVELOPMENT.md) - Developer setup guide
- [SECRETS_MANAGEMENT.md](../SECRETS_MANAGEMENT.md) - Security and secrets management
- [fastapi_app/app/_archive/README.md](../../fastapi_app/app/_archive/README.md) - Archived backend code (endpoints/modules), separate from this docs archive
