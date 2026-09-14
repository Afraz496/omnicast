# Omnicast

Automatic statistical forecasting for Python with one consistent, interval-aware API -- fit, backtest, and plot every model the same way.

> **Status:** v0.1 alpha (renamed from `auto-time-series`). The API is usable, but model coverage and R parity fixtures are still growing.

## Install

```bash
pip install omnicast
```

For local development:

```bash
uv sync --extra dev
```

Full docs with a worked example for every model, a real-data walkthrough, and the
complete API reference are hosted at **[afraz496.github.io/omnicast](https://afraz496.github.io/omnicast/)**
(source under [`docs/`](docs/index.md)). Build them locally with:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

Heavy dependencies (PyTorch, TimesFM) are kept out of the base install:

```bash
# For LSTMForecaster & DeepARForecaster (PyTorch):
pip install omnicast[torch]

# For TimesFMForecaster (Google TimesFM foundation model):
pip install omnicast[timesfm]

# For ChronosForecaster (Amazon Chronos foundation model):
pip install omnicast[chronos]

# Or install all optional dependencies:
pip install omnicast[all]

# For development:
uv sync --extra dev --extra torch --extra timesfm --extra chronos
```

## Quick start

```python
import pandas as pd
from omnicast import AutoForecaster

y = pd.Series(
    [112, 118, 121, 130, 128, 137, 143, 149, 154, 162, 169, 175],
    index=pd.period_range("2025-01", periods=12, freq="M"),
)

model = AutoForecaster(
    seasonal_period=None,
    metric="rmse",
    validation_horizon=1,
).fit(y)

forecast = model.predict(horizon=6, level=[80, 95])
print(model.leaderboard_)
print(forecast.to_frame())
```

Every fitted estimator exposes `fitted_values_`, `residuals_`, `sigma2_`, and `prediction_intervals_`. Statistical estimators also expose `params_`, `parameter_confidence_intervals_` (95%), `aic_`, and `bic_`. Prediction intervals are returned on each prediction because they depend on horizon and requested coverage.

## Models

| Estimator | Purpose | Intervals |
|---|---|---|
| `NaiveForecaster` | Random walk | Horizon-scaled Gaussian innovation |
| `SeasonalNaiveForecaster` | Seasonal random walk | Cycle-scaled Gaussian innovation |
| `MeanForecaster` | Historical mean | Mean forecast uncertainty |
| `DriftForecaster` | Random walk with drift | Drift forecast uncertainty |
| `ThetaForecaster` | Theta method (port of R `forecast::thetaf`) | Random-walk innovation scaling |
| `ETSForecaster` | Error/trend/seasonal state space | State-space forecast uncertainty |
| `ARIMAForecaster` | ARIMA/SARIMA, optional regressors | State-space forecast uncertainty |
| `AutoARIMAForecaster` | AICc grid-selected ARIMA | State-space forecast uncertainty |
| `LSTMForecaster` | Autoregressive LSTM (`torch`, optional) | Random-walk innovation scaling |
| `DeepARForecaster` | Probabilistic autoregressive RNN (`torch`, optional) | Monte Carlo sample trajectory scaling |
| `TimesFMForecaster` | Zero-shot foundation model (`timesfm`, optional) | Random-walk innovation scaling |
| `ChronosForecaster` | Zero-shot foundation model (`chronos`, optional) | Sample dispersion or residual scaling |
| `AutoForecaster` | Rolling-origin model selection | Selected model's intervals |

## Evaluation

```python
from omnicast import NaiveForecaster, backtest

folds = backtest(NaiveForecaster(), y, horizon=3, initial=6, metric="rmse")
print(folds)
```

Available metrics are MAE, RMSE, MAPE, and sMAPE. Backtesting uses expanding windows and never trains on future observations.

## Design and scope

The package follows pandas index semantics and the familiar `fit`/`predict` estimator pattern. Learned state uses trailing underscores. Models validate input rather than silently imputing data or guessing an irregular date frequency.

This codebase is a Python implementation foundation, not a blanket claim of parity with R forecasting packages. Each future port must record its algorithm source, licensing, deviations, and numerical parity tests. See [CONTRIBUTING.md](CONTRIBUTING.md).

`ThetaForecaster` is the first R port: a compatible pure-Python reimplementation of `forecast::thetaf`'s classical Theta method, described in its own docstring along with the exact deviations from R's output (approximate intervals, no numerical parity fixtures yet).

`LSTMForecaster` is the first wrapper around a Python deep-learning module (`torch`, optional dependency), following the same `BaseForecaster` interface as the statsmodels-backed models. It is not part of `AutoForecaster`'s default candidate list -- pass it explicitly via `AutoForecaster(models=[...])` -- since it is optional-dependency and materially slower to backtest.

`DeepARForecaster` implements the DeepAR probabilistic autoregressive recurrent network (Salinas et al., `torch`, optional dependency). Unlike point-forecast LSTMs, it models the parameters of a predictive distribution with Gaussian Negative Log-Likelihood and simulates Monte Carlo trajectory rollouts for calibrated intervals. Like other neural models, it is kept out of `AutoForecaster`'s default candidates.

`TimesFMForecaster` wraps Google Research's TimesFM zero-shot time-series foundation model (`timesfm`, optional dependency). By default it loads TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`), which is distributed under the Apache-2.0 license matching this project, and also supports TimesFM 3.0 checkpoints. Like `LSTMForecaster`, it is kept out of `AutoForecaster`'s default candidate list to preserve fast, lightweight local execution.

`ChronosForecaster` wraps Amazon Science's Chronos zero-shot time-series foundation models (`chronos-forecasting`, optional dependency). It converts time series into discrete tokens and predicts future steps autoregressively using pretrained transformer language models. Pretrained weights are Apache-2.0 licensed.

## Contributors

- **Afraz Arif Khan** ([@Afraz496](https://github.com/Afraz496)) -- core estimator API, statistical models, evaluation, and the Sphinx docs site.
- **Javier Martínez-Rodríguez** ([@JavierMtzRdz](https://github.com/JavierMtzRdz)) -- the plotting and backtesting system: `ForecastResult`/`BacktestResult` and their `.plot()` methods, the `Backtester` class, every function in `omnicast.plotting`, `AutoForecaster.plot_all()`, and the real-data forecasting walkthrough notebook.

Licensed under Apache-2.0.
