import numpy as np
import pandas as pd
import pytest

from omnicast import AutoForecaster, ChronosForecaster
from omnicast.models import chronos as chronos_module


def trending_series(n: int = 24) -> pd.Series:
    rng = np.random.default_rng(12)
    return pd.Series(
        10.0 + 0.4 * np.arange(n) + rng.normal(0, 0.1, n),
        index=pd.period_range("2024-01", periods=n, freq="M"),
    )


class MockSamplePipeline:
    """Mock for sample-based Chronos pipeline returning (1, num_samples, horizon)."""

    def predict(self, inputs: object, prediction_length: int, num_samples: int = 20):
        import torch

        arr = inputs.numpy() if hasattr(inputs, "numpy") else np.asarray(inputs, dtype=float)
        last_val = float(arr[-1])
        base_step = np.arange(1, prediction_length + 1) * 0.2
        samples = np.array([last_val + base_step + 0.02 * (s - num_samples / 2) for s in range(num_samples)])
        return torch.tensor(samples, dtype=torch.float32).unsqueeze(0)


class MockBoltPipeline:
    """Mock for Chronos-Bolt pipeline returning (1, num_quantiles, horizon)."""

    def predict(self, inputs: object, prediction_length: int):
        import torch

        arr = inputs.numpy() if hasattr(inputs, "numpy") else np.asarray(inputs, dtype=float)
        last_val = float(arr[-1])
        base = last_val + np.arange(1, prediction_length + 1) * 0.25
        # 9 quantiles [0.1, 0.2, ..., 0.9]
        quantiles = np.array([base + (q - 0.5) * 0.5 for q in np.linspace(0.1, 0.9, 9)])
        return torch.tensor(quantiles, dtype=torch.float32).unsqueeze(0)


def test_chronos_missing_dependency(monkeypatch):
    monkeypatch.setattr(chronos_module, "chronos", None)
    forecaster = ChronosForecaster()
    y = trending_series()
    with pytest.raises(ImportError, match="pip install omnicast\\[chronos\\]"):
        forecaster.fit(y)


def test_chronos_requires_minimum_observations():
    forecaster = ChronosForecaster(pipeline=MockSamplePipeline())
    with pytest.raises(ValueError, match="at least 3 observations"):
        forecaster.fit(pd.Series([1.0, 2.0]))


def test_chronos_sample_pipeline_forecast_and_intervals():
    y = trending_series()
    model = ChronosForecaster(pipeline=MockSamplePipeline(), num_samples=25)
    model.fit(y)
    res = model.predict(horizon=4, level=[80, 95])

    assert len(res.mean) == 4
    assert np.isfinite(res.mean).all()
    assert (res.lower[80] <= res.mean).all()
    assert (res.upper[80] >= res.mean).all()
    assert (res.upper[95] >= res.upper[80]).all()
    assert (res.lower[95] <= res.lower[80]).all()
    assert model.fitted_values_.index.equals(y.index)
    assert model.residuals_.index.equals(y.index)
    assert model.is_fitted_


def test_chronos_bolt_pipeline_forecast_and_intervals():
    y = trending_series()
    model = ChronosForecaster(pipeline=MockBoltPipeline())
    model.fit(y)
    res = model.predict(horizon=3, level=90)

    assert len(res.mean) == 3
    assert np.isfinite(res.mean).all()
    assert (res.lower[90] <= res.mean).all()
    assert (res.upper[90] >= res.mean).all()


def test_chronos_compute_fitted_false():
    y = trending_series()
    model = ChronosForecaster(pipeline=MockSamplePipeline(), compute_fitted=False)
    model.fit(y)
    assert np.isnan(model.fitted_values_.iloc[0])
    assert not np.isnan(model.fitted_values_.iloc[1])
    res = model.predict(horizon=2)
    assert len(res.mean) == 2


def test_chronos_works_with_auto_forecaster():
    y = trending_series(12)
    chronos_m = ChronosForecaster(pipeline=MockSamplePipeline(), compute_fitted=False)
    auto = AutoForecaster(models=[chronos_m], validation_horizon=2)
    auto.fit(y)
    res = auto.predict(horizon=2)

    assert len(res.mean) == 2
    assert auto.leaderboard_["model"].iloc[0] == "ChronosForecaster"
