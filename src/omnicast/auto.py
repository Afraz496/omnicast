"""Automatic selection across heterogeneous forecasting models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .evaluation import Backtester
from .models import (
    AutoARIMAForecaster,
    DriftForecaster,
    ETSForecaster,
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
)
from .utils import validate_series


class AutoForecaster(BaseForecaster):
    """Choose the model with the best rolling-origin validation score.

    With ``keep_all=True`` every candidate is also fitted on the full series and
    kept in ``fitted_`` (name -> fitted model).

    Examples
    --------
    >>> import pandas as pd
    >>> from omnicast import AutoForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    >>> model = AutoForecaster().fit(y)
    >>> model.leaderboard_["model"].iloc[0]
    'ThetaForecaster'
    >>> model.predict(horizon=2).mean.round(2).tolist()
    [14.05, 14.49]

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - The default entry point -- backtests a panel of candidates and
           refits the winner, so you don't have to pick a model by hand
       * - Avoid when
         - You already know which model fits, need exogenous regressors
           (not yet supported here), or want a single deterministic model
           without a backtest step
       * - Handles trend
         - Depends on which candidate wins the backtest
       * - Handles seasonality
         - Yes, via ``seasonal_period`` (adds seasonal-aware candidates)
       * - Extra dependencies
         - None by default; include :class:`~omnicast.LSTMForecaster`
           explicitly via ``models=[...]`` to pull in ``torch``
       * - Min. observations
         - Whatever the strictest candidate in ``models`` requires; a
           failing candidate is skipped rather than aborting selection
    """

    def __init__(
        self,
        models: list[BaseForecaster] | None = None,
        seasonal_period: int | None = None,
        metric: str = "rmse",
        validation_horizon: int = 1,
        keep_all: bool = False,
        as_ensemble: bool = False,
        weights: Sequence[float] | dict[str, float] | np.ndarray | None = None,
        **kwargs,
    ):
        self.models = models
        self.seasonal_period = seasonal_period
        self.metric = metric
        self.validation_horizon = validation_horizon
        self.keep_all = keep_all
        self.as_ensemble = as_ensemble or bool(kwargs.get("ensemble", False))
        self.weights = weights

    def fit(self, y, X=None):
        if X is not None:
            raise NotImplementedError(
                "AutoForecaster exogenous model selection is not yet supported"
            )
        series = validate_series(y)
        candidates = list(self.models) if self.models else [
            NaiveForecaster(),
            MeanForecaster(),
            DriftForecaster(),
            ThetaForecaster(),
            ETSForecaster(),
            AutoARIMAForecaster(seasonal_period=self.seasonal_period),
        ]
        if self.seasonal_period and len(series) > self.seasonal_period:
            candidates.insert(1, SeasonalNaiveForecaster(self.seasonal_period))
        rows, successful = [], []
        initial = max(5, len(series) - max(3 * self.validation_horizon, len(series) // 4))
        bt = Backtester(horizon=self.validation_horizon, initial=initial, metric=self.metric)
        for model in candidates:
            try:
                score = float(bt.run(model, series)["score"].mean())
                rows.append({"model": type(model).__name__, "score": score, "status": "ok"})
                successful.append((score, model))
            # Candidate libraries expose heterogeneous numerical failure types;
            # one failed candidate must not prevent selection among the rest.
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {"model": type(model).__name__, "score": float("inf"), "status": str(exc)}
                )
        if not successful:
            raise RuntimeError("All candidate models failed")
        self.candidates_ = [model for _, model in successful]
        self.leaderboard_ = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
        best = min(successful, key=lambda item: item[0])[1]
        if self.keep_all:
            self.fitted_ = {
                type(m).__name__: deepcopy(m).fit(series) for m in self.candidates_
            }
            self.best_model_ = self.fitted_[type(best).__name__]
        else:
            self.fitted_ = None
            self.best_model_ = deepcopy(best).fit(series)
        self.y_, self.is_fitted_, self.n_obs_ = series, True, len(series)
        if self.as_ensemble:
            self.ensemble_model_ = self.to_ensemble(weights=self.weights)
            self.best_model_ = self.ensemble_model_
            self.fitted_values_ = self.best_model_.fitted_values_
            self.residuals_, self.sigma2_ = self.best_model_.residuals_, self.best_model_.sigma2_
        else:
            self.fitted_values_ = self.best_model_.fitted_values_
            self.residuals_, self.sigma2_ = self.best_model_.residuals_, self.best_model_.sigma2_
            if hasattr(self.best_model_, "parameter_confidence_intervals_"):
                self.parameter_confidence_intervals_ = (
                    self.best_model_.parameter_confidence_intervals_
                )
        return self

    def predict(self, horizon, X=None, level=(80, 95)):
        self._check_fitted()
        result = self.best_model_.predict(horizon, X=X, level=level)
        self.forecast_, self.prediction_intervals_ = result, self.best_model_.prediction_intervals_
        return result

    def plot_all(
        self, horizon=None, level=(80, 95), observed=None, intervals=False, **kwargs
    ):
        """Overlay every successful candidate's forecast on one axes.

        Refits each candidate on the full training series, unless the estimator
        was built with ``keep_all=True`` (then the stored fits are reused).

        Parameters
        ----------
        horizon : int, optional
            Steps to forecast. Defaults to ``validation_horizon``.
        level : float or sequence of float, default (80, 95)
            Prediction-interval coverage levels to compute.
        observed : pd.Series, optional
            History to draw. Defaults to the full training series.
        intervals : bool, default False
            Shade each candidate's prediction bands. Off by default -- with
            several candidates the overlapping bands get muddy.
        **kwargs
            Forwarded to the trajectory plot (`title`, `ax`, `save_path`, ...).
        """
        from .plotting import plot_forecast_trajectories

        self._check_fitted()
        horizon = horizon or self.validation_horizon
        if self.fitted_ is not None:
            forecasts = {n: m.predict(horizon, level=level) for n, m in self.fitted_.items()}
        else:
            forecasts = {
                type(m).__name__: deepcopy(m).fit(self.y_).predict(horizon, level=level)
                for m in self.candidates_
            }
        return plot_forecast_trajectories(
            forecasts,
            observed=self.y_ if observed is None else observed,
            intervals=intervals,
            **kwargs,
        )

    def to_ensemble(
        self,
        weights: Sequence[float] | dict[str, float] | np.ndarray | None = None,
        top_k: int | None = None,
    ) -> EnsembleForecaster:
        """Create an EnsembleForecaster from candidate models.

        Parameters
        ----------
        weights : sequence of float, dict, or np.ndarray, optional
            Weights assigned to each forecaster. Defaults to None, giving equal
            weight (1 / N) to all selected forecasters.
        top_k : int, optional
            Number of top-performing models from the leaderboard to include
            in the ensemble. If None (default), all successful candidates are included.

        Returns
        -------
        EnsembleForecaster
            A fitted ensemble forecaster.
        """
        self._check_fitted()
        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be at least 1")
            top_names = list(self.leaderboard_["model"].head(top_k))
            selected_models = [m for m in self.candidates_ if type(m).__name__ in top_names]
        else:
            selected_models = list(self.candidates_)

        ens = EnsembleForecaster(models=selected_models, weights=weights)
        if (
            self.fitted_ is not None
            and top_k is None
            and all(type(m).__name__ in self.fitted_ for m in selected_models)
        ):
            ens.y_ = self.y_
            ens.fitted_models_ = [self.fitted_[type(m).__name__] for m in selected_models]
            n = len(ens.fitted_models_)
            if weights is None:
                raw_w = np.full(n, 1.0 / n, dtype=float)
            elif isinstance(weights, dict):
                raw_w = np.zeros(n, dtype=float)
                for i, m in enumerate(ens.fitted_models_):
                    name = type(m).__name__
                    if name in weights:
                        raw_w[i] = float(weights[name])
                    elif i in weights:
                        raw_w[i] = float(weights[i])
                    else:
                        raise ValueError(f"Weight missing for model '{name}'")
            else:
                raw_w = np.asarray(weights, dtype=float)
                if len(raw_w) != n:
                    raise ValueError(
                        f"weights length ({len(raw_w)}) must match number of models ({n})"
                    )
            if np.any(raw_w < 0) or np.sum(raw_w) <= 0:
                raise ValueError("Weights must be non-negative with strictly positive sum")
            ens.weights_ = raw_w / np.sum(raw_w)
            ens.weights_dict_ = {
                type(m).__name__: float(w) for m, w in zip(ens.fitted_models_, ens.weights_)
            }
            fitted_vals = ens._fitted_values()
            ens.fitted_values_ = pd.Series(fitted_vals, index=self.y_.index, name="fitted")
            ens.residuals_ = (self.y_ - ens.fitted_values_).rename("residual")
            finite = ens.residuals_.dropna()
            ens.sigma2_ = float(np.mean(np.square(finite))) if len(finite) else 0.0
            ens.n_obs_ = len(self.y_)
            ens.is_fitted_ = True
            return ens

        return ens.fit(self.y_)

    ensemble = to_ensemble

    def predict_ensemble(
        self,
        horizon: int,
        weights: Sequence[float] | dict[str, float] | np.ndarray | None = None,
        level: float | Iterable[float] = (80, 95),
        top_k: int | None = None,
    ):
        """Forecast using an ensemble of the selected candidate models."""
        self._check_fitted()
        ens = self.to_ensemble(weights=weights, top_k=top_k)
        return ens.predict(horizon, level=level)

    def _fit(self, y):
        pass

    def _fitted_values(self):
        return self.best_model_.fitted_values_.to_numpy()

    def _forecast(self, horizon):
        return self.best_model_._forecast(horizon)


class EnsembleForecaster(BaseForecaster):
    """Combine forecasts from multiple models using equal or custom weights.

    By default, all selected forecasters receive equal weights (1 / N).
    Custom weights can be specified via the ``weights`` parameter.

    Parameters
    ----------
    models : list[BaseForecaster] | None, default None
        Forecasters to combine. If None, uses a default candidate suite:
        ``NaiveForecaster``, ``MeanForecaster``, ``DriftForecaster``,
        ``ThetaForecaster``, ``ETSForecaster``, and ``AutoARIMAForecaster``.
    weights : sequence of float, dict of {str: float}, or np.ndarray, optional
        Weights assigned to each forecaster. If None (default), assigns equal
        weight (1 / N) to all forecasters. If a sequence or array, its length
        must match ``models``. If a dictionary, keys can be model class names
        (e.g. ``{"ThetaForecaster": 0.6, "ETSForecaster": 0.4}``) or model
        indices. All weights must be non-negative and sum to a positive value;
        they are automatically normalized to sum to 1.0.
    seasonal_period : int, optional
        Used when ``models`` is None to configure seasonal candidates
        (e.g., SeasonalNaiveForecaster and seasonal AutoARIMA).

    Examples
    --------
    >>> import pandas as pd
    >>> from omnicast import EnsembleForecaster, MeanForecaster, NaiveForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    >>> model = EnsembleForecaster(models=[NaiveForecaster(), MeanForecaster()]).fit(y)
    >>> model.predict(horizon=2).mean.round(2).tolist()
    [13.25, 13.25]
    >>> weighted = EnsembleForecaster(
    ...     models=[NaiveForecaster(), MeanForecaster()], weights=[0.75, 0.25]
    ... ).fit(y)
    >>> weighted.predict(horizon=2).mean.round(2).tolist()
    [13.62, 13.62]

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - Hedging model-risk and improving forecast stability by combining
           complementary forecasters into a single prediction
       * - Avoid when
         - A single fast baseline or transparent individual model is strictly
           required, or computational resources cannot fit multiple candidates
       * - Handles trend
         - Yes, if one or more component models handle trend
       * - Handles seasonality
         - Yes, if one or more component models handle seasonality
       * - Extra dependencies
         - None by default; depends on the component models supplied
       * - Min. observations
         - Determined by the strictest component model in ``models``
    """

    def __init__(
        self,
        models: list[BaseForecaster] | None = None,
        weights: Sequence[float] | dict[str, float] | np.ndarray | None = None,
        seasonal_period: int | None = None,
    ):
        self.models = models
        self.weights = weights
        self.seasonal_period = seasonal_period

    def _fit(self, y: pd.Series) -> None:
        if self.models is not None:
            candidates = [deepcopy(m) for m in self.models]
        else:
            candidates = [
                NaiveForecaster(),
                MeanForecaster(),
                DriftForecaster(),
                ThetaForecaster(),
                ETSForecaster(),
                AutoARIMAForecaster(seasonal_period=self.seasonal_period),
            ]
            if self.seasonal_period and len(y) > self.seasonal_period:
                candidates.insert(1, SeasonalNaiveForecaster(self.seasonal_period))

        if not candidates:
            raise ValueError("models list cannot be empty")

        fitted_models = [deepcopy(m).fit(y) for m in candidates]
        self.fitted_models_ = fitted_models
        n_models = len(fitted_models)

        if self.weights is None:
            raw_weights = np.full(n_models, 1.0 / n_models, dtype=float)
        elif isinstance(self.weights, dict):
            raw_weights = np.zeros(n_models, dtype=float)
            for i, m in enumerate(fitted_models):
                name = type(m).__name__
                if name in self.weights:
                    raw_weights[i] = float(self.weights[name])
                elif i in self.weights:
                    raw_weights[i] = float(self.weights[i])
                else:
                    raise ValueError(
                        f"Weight not provided for model '{name}' (index {i}) in weights dictionary"
                    )
        else:
            try:
                raw_weights = np.asarray(self.weights, dtype=float)
            except Exception as e:
                raise ValueError(f"Could not convert weights to float array: {e}") from e
            if raw_weights.ndim != 1 or len(raw_weights) != n_models:
                raise ValueError(
                    f"weights length ({len(raw_weights)}) must match number of models ({n_models})"
                )

        if np.any(raw_weights < 0):
            raise ValueError("All weights must be non-negative")
        total_w = float(np.sum(raw_weights))
        if total_w <= 0:
            raise ValueError("Sum of weights must be strictly positive")

        self.weights_ = raw_weights / total_w
        self.weights_dict_ = {
            type(m).__name__: float(w) for m, w in zip(self.fitted_models_, self.weights_)
        }

    def _fitted_values(self) -> np.ndarray:
        fitted_matrix = np.column_stack(
            [m.fitted_values_.to_numpy() for m in self.fitted_models_]
        )
        w = self.weights_
        valid = ~np.isnan(fitted_matrix)
        masked_w = valid * w
        sum_w = np.sum(masked_w, axis=1)
        safe_matrix = np.where(valid, fitted_matrix, 0.0)
        weighted_sum = np.sum(safe_matrix * w, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(sum_w > 0, weighted_sum / sum_w, np.nan)
        return result

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        means = []
        ses = []
        for model in self.fitted_models_:
            m_mean, m_se = model._forecast(horizon)
            means.append(np.asarray(m_mean, dtype=float))
            ses.append(np.asarray(m_se, dtype=float))

        means_arr = np.array(means)
        ses_arr = np.array(ses)
        w = self.weights_[:, np.newaxis]

        ens_mean = np.sum(w * means_arr, axis=0)
        var_within = np.sum(w * np.square(ses_arr), axis=0)
        var_between = np.sum(w * np.square(means_arr - ens_mean), axis=0)
        ens_var = var_within + var_between
        ens_se = np.sqrt(np.maximum(ens_var, 0.0))

        return ens_mean, ens_se

    def plot_components(
        self,
        horizon: int | None = None,
        level: float | Iterable[float] = (80, 95),
        observed: pd.Series | None = None,
        intervals: bool = False,
        **kwargs,
    ):
        """Plot component candidate forecasts alongside the ensemble forecast.

        Parameters
        ----------
        horizon : int, optional
            Steps to forecast. Defaults to 1.
        level : float or sequence of float, default (80, 95)
            Prediction-interval coverage levels.
        observed : pd.Series, optional
            History to draw. Defaults to the training series.
        intervals : bool, default False
            Whether to shade individual component interval bands.
        **kwargs
            Forwarded to ``plot_forecast_trajectories``.
        """
        from .plotting import plot_forecast_trajectories

        self._check_fitted()
        h = horizon or 1
        forecasts = {}
        for m in self.fitted_models_:
            name = type(m).__name__
            if name in forecasts:
                idx = 2
                while f"{name}_{idx}" in forecasts:
                    idx += 1
                name = f"{name}_{idx}"
            forecasts[name] = m.predict(h, level=level)
        forecasts["Ensemble"] = self.predict(h, level=level)
        return plot_forecast_trajectories(
            forecasts,
            observed=self.y_ if observed is None else observed,
            intervals=intervals,
            **kwargs,
        )

