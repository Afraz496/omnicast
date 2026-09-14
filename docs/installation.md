# Installation

```bash
pip install omnicast
```

Heavy dependencies are kept out of the base install because they are large and optional:

```bash
# PyTorch for LSTMForecaster and DeepARForecaster
pip install omnicast[torch]

# Google TimesFM foundation model
pip install omnicast[timesfm]

# Amazon Chronos foundation model
pip install omnicast[chronos]

# All optional dependencies
pip install omnicast[all]
```

## Local development

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --extra dev                          # core + test/lint tooling
uv sync --extra dev --extra torch            # also install LSTM & DeepAR dependencies
uv sync --extra dev --extra timesfm          # also install TimesFM's dependency
uv sync --extra dev --extra chronos          # also install Chronos's dependency
uv sync --extra docs                         # sphinx + theme, to build this site
```

Run the test suite and linter:

```bash
uv run pytest
uv run ruff check .
```

Build these docs:

```bash
uv run sphinx-build -b html docs docs/_build/html
```
