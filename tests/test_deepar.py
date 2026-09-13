import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from omnicast import AutoForecaster, DeepARForecaster
from omnicast.models import deepar as deepar_module


def trending_series(n: int = 30) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(
        10.0 + 0.3 * np.arange(n) + rng.normal(0, 0.2, n),
        index=pd.period_range("2020-01", periods=n, freq="M"),
    )


def test_deepar_forecast_shape_and_intervals():
    y = trending_series()
    model = DeepARForecaster(lookback=6, hidden_size=8, num_layers=1, epochs=25, seed=0)
    forecast = model.fit(y).predict(4, level=[80, 95])

    assert len(forecast.mean) == 4
    assert np.isfinite(forecast.mean).all()
    assert (forecast.lower[80] <= forecast.mean).all()
    assert (forecast.upper[80] >= forecast.mean).all()
    assert (forecast.upper[95] >= forecast.upper[80]).all()
    assert (forecast.lower[95] <= forecast.lower[80]).all()


def test_deepar_fitted_values_align_with_input_index():
    y = trending_series()
    model = DeepARForecaster(lookback=6, hidden_size=8, num_layers=1, epochs=25, seed=0)
    model.fit(y)

    assert model.fitted_values_.index.equals(y.index)
    assert model.fitted_values_.iloc[:6].isna().all()
    assert np.isfinite(model.fitted_values_.iloc[6:].to_numpy()).all()
    assert model.residuals_.index.equals(y.index)
    assert model.is_fitted_


def test_deepar_requires_minimum_observations():
    with pytest.raises(ValueError, match="at least 14 observations"):
        DeepARForecaster(lookback=12).fit(pd.Series(np.arange(10.0)))


def test_deepar_import_error_when_torch_missing(monkeypatch):
    monkeypatch.setattr(deepar_module, "torch", None)
    forecaster = DeepARForecaster(lookback=4)
    with pytest.raises(ImportError, match="pip install omnicast\\[torch\\]"):
        forecaster.fit(pd.Series(np.arange(10.0)))


def test_deepar_works_with_auto_forecaster():
    y = trending_series(16)
    deepar = DeepARForecaster(lookback=4, hidden_size=8, num_layers=1, epochs=15, seed=0)
    auto = AutoForecaster(models=[deepar], validation_horizon=2)
    auto.fit(y)
    res = auto.predict(horizon=2)

    assert len(res.mean) == 2
    assert auto.leaderboard_["model"].iloc[0] == "DeepARForecaster"
