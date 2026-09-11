import numpy as np
import pandas as pd
import pytest

from omnicast import (
    AutoForecaster,
    DriftForecaster,
    EnsembleForecaster,
    ForecastResult,
    MeanForecaster,
    NaiveForecaster,
    ThetaForecaster,
    backtest,
)


def test_ensemble_default_models_and_equal_weights():
    y = pd.Series(np.arange(20.0))
    model = EnsembleForecaster()
    model.fit(y)
    assert model.is_fitted_
    n = len(model.fitted_models_)
    assert n >= 5
    assert np.allclose(model.weights_, 1.0 / n)
    for w in model.weights_dict_.values():
        assert np.isclose(w, 1.0 / n)

    forecast = model.predict(horizon=3)
    assert isinstance(forecast, ForecastResult)
    assert len(forecast.mean) == 3


def test_ensemble_custom_weights_list():
    y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    m_naive = NaiveForecaster().fit(y)
    m_mean = MeanForecaster().fit(y)
    p_naive = m_naive.predict(2).mean
    p_mean = m_mean.predict(2).mean

    # Explicit list weights
    ens = EnsembleForecaster(models=[NaiveForecaster(), MeanForecaster()], weights=[0.8, 0.2]).fit(y)
    assert np.allclose(ens.weights_, [0.8, 0.2])
    expected = 0.8 * p_naive + 0.2 * p_mean
    assert np.allclose(ens.predict(2).mean, expected)

    # Unnormalized weights should be auto-normalized
    ens_unnorm = EnsembleForecaster(
        models=[NaiveForecaster(), MeanForecaster()], weights=[4.0, 1.0]
    ).fit(y)
    assert np.allclose(ens_unnorm.weights_, [0.8, 0.2])
    assert np.allclose(ens_unnorm.predict(2).mean, expected)


def test_ensemble_custom_weights_dict():
    y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    ens = EnsembleForecaster(
        models=[NaiveForecaster(), MeanForecaster()],
        weights={"NaiveForecaster": 3.0, "MeanForecaster": 1.0},
    ).fit(y)
    assert np.isclose(ens.weights_dict_["NaiveForecaster"], 0.75)
    assert np.isclose(ens.weights_dict_["MeanForecaster"], 0.25)


def test_ensemble_invalid_weights_raise():
    y = pd.Series(np.arange(10.0))
    # Length mismatch
    with pytest.raises(ValueError, match="weights length"):
        EnsembleForecaster(models=[NaiveForecaster(), MeanForecaster()], weights=[0.5]).fit(y)

    # Negative weights
    with pytest.raises(ValueError, match="non-negative"):
        EnsembleForecaster(models=[NaiveForecaster(), MeanForecaster()], weights=[1.5, -0.5]).fit(y)

    # Sum is zero
    with pytest.raises(ValueError, match="positive"):
        EnsembleForecaster(models=[NaiveForecaster(), MeanForecaster()], weights=[0.0, 0.0]).fit(y)

    # Missing model in dict
    with pytest.raises(ValueError, match="Weight not provided for model"):
        EnsembleForecaster(
            models=[NaiveForecaster(), MeanForecaster()], weights={"NaiveForecaster": 1.0}
        ).fit(y)

    # Empty models list
    with pytest.raises(ValueError, match="models list cannot be empty"):
        EnsembleForecaster(models=[]).fit(y)


def test_ensemble_fitted_values_and_residuals():
    y = pd.Series(np.arange(15.0) + 1.0)
    ens = EnsembleForecaster(models=[NaiveForecaster(), MeanForecaster()]).fit(y)
    assert len(ens.fitted_values_) == len(y)
    assert len(ens.residuals_) == len(y)
    assert ens.sigma2_ >= 0.0


def test_ensemble_prediction_intervals():
    y = pd.Series(np.arange(20.0) + np.random.default_rng(0).normal(0, 0.5, 20))
    ens = EnsembleForecaster(models=[NaiveForecaster(), DriftForecaster(), MeanForecaster()]).fit(y)
    res = ens.predict(horizon=4, level=(80, 95))
    int80 = res.interval(80)
    int95 = res.interval(95)

    # 95% interval must enclose the 80% interval, which encloses the mean
    assert (int95["lower"] <= int80["lower"]).all()
    assert (int80["lower"] <= res.mean).all()
    assert (res.mean <= int80["upper"]).all()
    assert (int80["upper"] <= int95["upper"]).all()


def test_ensemble_with_backtest():
    y = pd.Series(np.arange(20.0))
    ens = EnsembleForecaster(models=[NaiveForecaster(), DriftForecaster()])
    res = backtest(ens, y, horizon=2, initial=12, metric="rmse")
    assert len(res) == 7
    assert (res["score"] >= 0).all()


def test_auto_forecaster_ensemble_method():
    y = pd.Series(np.arange(20.0))
    auto = AutoForecaster(
        models=[NaiveForecaster(), DriftForecaster(), MeanForecaster()],
        validation_horizon=2,
    ).fit(y)

    # auto.ensemble() gives equal weight to all selected forecasters by default
    ens = auto.ensemble()
    assert isinstance(ens, EnsembleForecaster)
    assert len(ens.fitted_models_) == 3
    assert np.allclose(ens.weights_, 1.0 / 3.0)

    # auto.predict_ensemble() produces identical predictions to ens.predict()
    p_ens1 = auto.predict_ensemble(horizon=3)
    p_ens2 = ens.predict(horizon=3)
    assert np.allclose(p_ens1.mean, p_ens2.mean)

    # auto.ensemble with top_k
    ens_top2 = auto.ensemble(top_k=2)
    assert len(ens_top2.fitted_models_) == 2
    assert np.allclose(ens_top2.weights_, 0.5)

    # auto.ensemble with custom weights
    ens_custom = auto.ensemble(weights=[0.6, 0.3, 0.1])
    assert np.allclose(ens_custom.weights_, [0.6, 0.3, 0.1])


def test_auto_forecaster_keep_all_ensemble():
    y = pd.Series(np.arange(20.0))
    auto = AutoForecaster(
        models=[NaiveForecaster(), MeanForecaster()],
        keep_all=True,
    ).fit(y)
    ens = auto.ensemble()
    assert ens.is_fitted_
    assert len(ens.fitted_models_) == 2
    assert np.allclose(ens.weights_, 0.5)


def test_auto_forecaster_ensemble_flag():
    y = pd.Series(np.arange(20.0))
    # If ensemble=True on AutoForecaster, predict delegates to the ensemble
    auto = AutoForecaster(
        models=[NaiveForecaster(), DriftForecaster(), MeanForecaster()],
        ensemble=True,
    ).fit(y)
    assert hasattr(auto, "ensemble_model_")
    assert isinstance(auto.ensemble_model_, EnsembleForecaster)
    pred = auto.predict(horizon=2)
    assert len(pred.mean) == 2


def test_ensemble_plot_components():
    import matplotlib
    matplotlib.use("Agg")

    y = pd.Series(np.arange(15.0))
    ens = EnsembleForecaster(models=[NaiveForecaster(), ThetaForecaster()]).fit(y)
    ax = ens.plot_components(horizon=2)
    assert ax is not None
