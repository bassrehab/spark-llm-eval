# Contributing to spark-llm-eval

Thanks for your interest in contributing! This document outlines how to get started.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/bassrehab/spark-llm-eval.git
cd spark-llm-eval
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests:
```bash
pytest tests/unit/ -v
```

## Code Style

- Python 3.10+
- Type hints for public APIs
- Google-style docstrings for public functions
- Format with Black (default settings)
- Lint with Ruff

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest tests/unit/`)
5. Commit with a clear message
6. Push and open a Pull Request

## Adding New Metrics

1. Create your metric class in the appropriate file under `spark_llm_eval/evaluation/`
2. Inherit from `Metric` base class
3. Implement the `compute()` method
4. Register with `@register_metric` decorator
5. Add unit tests in `tests/unit/`

## Adding New Inference Providers

1. Create a new file in `spark_llm_eval/inference/`
2. Inherit from `InferenceEngine`
3. Implement `initialize()`, `infer()`, `infer_batch()`, `shutdown()`
4. Add to `ModelProvider` enum in `core/config.py`
5. Register in the engine factory
6. Add tests

## Questions?

Open an issue on GitHub or reach out to the maintainers.
