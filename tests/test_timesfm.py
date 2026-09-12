import numpy as np
import pandas as pd
import pytest

from omnicast import AutoForecaster, TimesFMForecaster
from omnicast.models import timesfm as timesfm_module


def trending_series(n: int = 24) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(
        10.0 + 0.5 * np.arange(n) + rng.normal(0, 0.1, n),
        index=pd.period_range("2024-01", periods=n, freq="M"),
    )


class MockTimesFM2p5:
    """Mock for TimesFM 2.5 torch backend with .forecast()."""

    def forecast(
        self, horizon: int, inputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(inputs[0], dtype=float)
        last_val = arr[-1]
        preds = np.array([[last_val + 0.2 * (i + 1) for i in range(horizon)]])
        quantiles = np.zeros((1, horizon, 10))
        return preds, quantiles


class MockTimesFM3:
    """Mock for TimesFM 3 backend with .predict()."""

    def predict(self, context: np.ndarray, horizon: int, **kwargs: object):
        from dataclasses import dataclass

        @dataclass
        class MockForecastOutput:
            forecast: np.ndarray
            quantiles: np.ndarray | None = None

        last_val = float(context[-1])
        preds = np.array([last_val + 0.3 * (i + 1) for i in range(horizon)])
        return MockForecastOutput(forecast=preds)


def test_timesfm_import_error_when_dependency_missing(monkeypatch):
    monkeypatch.setattr(timesfm_module, "timesfm", None)
    forecaster = TimesFMForecaster()
    y = trending_series()
    with pytest.raises(ImportError, match="pip install omnicast\\[timesfm\\]"):
        forecaster.fit(y)


def test_timesfm_requires_minimum_observations():
    model = TimesFMForecaster(model=MockTimesFM2p5())
    with pytest.raises(ValueError, match="at least 3 observations"):
        model.fit(pd.Series([1.0, 2.0]))


def test_timesfm_2p5_forecast_shape_and_intervals():
    y = trending_series()
    forecaster = TimesFMForecaster(model=MockTimesFM2p5())
    forecaster.fit(y)
    res = forecaster.predict(horizon=4, level=[80, 95])

    assert len(res.mean) == 4
    assert np.isfinite(res.mean).all()
    assert res.mean.index[0] == y.index[-1] + 1
    assert (res.lower[80] <= res.mean).all()
    assert (res.upper[80] >= res.mean).all()
    assert (res.upper[95] >= res.upper[80]).all()
    assert (res.lower[95] <= res.lower[80]).all()
    assert forecaster.fitted_values_.index.equals(y.index)
    assert forecaster.residuals_.index.equals(y.index)
    assert forecaster.is_fitted_
    assert forecaster.sigma2_ >= 0


def test_timesfm_3_forecast_shape_and_intervals():
    y = trending_series()
    forecaster = TimesFMForecaster(model=MockTimesFM3())
    forecaster.fit(y)
    res = forecaster.predict(horizon=5, level=90)

    assert len(res.mean) == 5
    assert np.isfinite(res.mean).all()
    assert (res.lower[90] <= res.mean).all()
    assert (res.upper[90] >= res.mean).all()


def test_timesfm_compute_fitted_false():
    y = trending_series()
    forecaster = TimesFMForecaster(model=MockTimesFM2p5(), compute_fitted=False)
    forecaster.fit(y)
    assert np.isnan(forecaster.fitted_values_.iloc[0])
    assert not np.isnan(forecaster.fitted_values_.iloc[1])
    res = forecaster.predict(horizon=3)
    assert len(res.mean) == 3


def test_timesfm_works_with_auto_forecaster():
    y = trending_series(12)
    tfm = TimesFMForecaster(model=MockTimesFM2p5(), compute_fitted=False)
    auto = AutoForecaster(models=[tfm], validation_horizon=2)
    auto.fit(y)
    res = auto.predict(horizon=2)
    assert len(res.mean) == 2
    assert auto.leaderboard_["model"].iloc[0] == "TimesFMForecaster"
