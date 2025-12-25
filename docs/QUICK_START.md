# Quick Start Guide - pyproject.toml Setup

## Installation

### Development Setup (with tools)
```bash
cd fastapi_app
pip install -e ".[dev]"
```

### Production Setup
```bash
cd fastapi_app
pip install .
```

## Docker

### Build and Run
```bash
cd fastapi_app
docker-compose up -d
```

### Check Health
```bash
curl http://localhost:8000/health
```

### View Logs
```bash
docker-compose logs -f
```

### Stop
```bash
docker-compose down
```

## Code Quality

### Format Code
```bash
cd fastapi_app
black .
```

### Lint Code
```bash
cd fastapi_app
ruff check .
ruff check --fix .  # Auto-fix issues
```

### Type Check
```bash
cd fastapi_app
mypy app/
```

### Run All Quality Checks
```bash
cd fastapi_app
black --check .
ruff check .
mypy app/
```

## Testing

### Run Tests
```bash
cd fastapi_app
pytest
```

### Run with Coverage
```bash
cd fastapi_app
pytest --cov
```

### Run Specific Tests
```bash
cd fastapi_app
pytest tests/test_api.py              # Specific file
pytest tests/test_api.py::test_health  # Specific test
pytest -m unit                         # Unit tests only
pytest -m integration                  # Integration tests only
```

## Pre-commit Hooks

### Install Hooks
```bash
# From project root
pre-commit install
```

### Run Manually
```bash
# From project root
pre-commit run --all-files
```

## Adding Dependencies

### 1. Edit pyproject.toml
```toml
[project]
dependencies = [
    "new-package>=1.0.0",
]
```

### 2. Update requirements.txt
```bash
cd fastapi_app
pip install -e ".[dev]"
pip freeze > requirements.txt
```

### 3. Test
```bash
docker-compose build
docker-compose up -d
```

## Common Commands

### Check Installation
```bash
cd fastapi_app
python -c "import tomllib; print('✓ pyproject.toml valid')"
```

### View Installed Packages
```bash
pip list
```

### Check Tool Versions
```bash
black --version
ruff --version
pytest --version
mypy --version
```

## Troubleshooting

### Import Errors
```bash
pip install -e ".[dev]"
```

### Docker Issues
```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Tool Config Issues
Make sure you're in the fastapi_app directory where pyproject.toml is located.

## Documentation

- **Full Development Guide:** [DEVELOPMENT.md](DEVELOPMENT.md)
- **Migration Summary:** [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)
- **Main README:** [../README.md](../README.md)

---

For detailed information, see [DEVELOPMENT.md](DEVELOPMENT.md).
