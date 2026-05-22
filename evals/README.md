# JSE Chatbot Eval Suite

Persona-driven, multi-turn simulation suite for the JSE DataSphere chatbot.
See [spec](../docs/superpowers/specs/2026-05-23-simulation-eval-design.md)
for design rationale.

## Quick start

```bash
cd evals
pip install -e ".[dev]"
pytest                                 # run unit tests
python -m evals.cli --help             # see runner options
```

Full usage docs in this README will be filled in as the suite is built.
