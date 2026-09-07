"""Statsmodels-backed statistical forecasters."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..base import BaseForecaster


class ETSForecaster(BaseForecaster):
    """Error-trend-seasonal state-space model with automatic component defaults.

    Examples
    --------
    >>> import pandas as pd
    >>> from omnicast import ETSForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    >>> model = ETSForecaster(trend="add").fit(y)
    >>> model.predict(horizon=2).mean.round(2).tolist()
    [15.6, 16.49]

    Pass ``seasonal="add"`` (or ``"mul"``) with ``seasonal_period`` set to
    model a repeating cycle alongside the trend.

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - Series with a known trend/seasonal shape where you want a full
           statistical fit (AIC/BIC, parameter confidence intervals) rather
           than an approximation
       * - Avoid when
         - You want the trend/seasonal order searched for you (see
           :class:`AutoARIMAForecaster`) or need exogenous regressors
       * - Handles trend
         - Yes, via ``trend`` (``"add"``, ``"mul"``, or ``None``)
       * - Handles seasonality
         - Yes, via ``seasonal`` + ``seasonal_period``
       * - Extra dependencies
         - None (uses ``statsmodels``)
       * - Min. observations
         - Enough for ``statsmodels`` to estimate the requested
           components; at least ``2 * seasonal_period`` when seasonal
    """

    def __init__(
        self,
        trend: str | None = "add",
        seasonal: str | None = None,
        seasonal_period: int | None = None,
        damped_trend: bool = False,
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.damped_trend = damped_trend

    def _fit(self, y: pd.Series) -> None:
        if self.seasonal is not None and not self.seasonal_period:
            raise ValueError("seasonal_period is required when seasonal is enabled")
        model = ETSModel(
            y,
            error="add",
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_period,
        )
        self.result_ = model.fit(disp=False)
        self.params_ = self.result_.params.copy()
        self.parameter_confidence_intervals_ = self.result_.conf_int(alpha=0.05)
        self.aic_ = float(self.result_.aic)
        self.bic_ = float(self.result_.bic)

    def _fitted_values(self) -> np.ndarray:
        return np.asarray(self.result_.fittedvalues)

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.result_.get_prediction(start=len(self.y_), end=len(self.y_) + horizon - 1)
        mean = np.asarray(prediction.predicted_mean)
        se = np.sqrt(np.asarray(prediction.var_pred_mean))
        return mean, se


class ARIMAForecaster(BaseForecaster):
    """ARIMA/SARIMA model with optional exogenous regressors.

    Examples
    --------
    >>> import pandas as pd
    >>> from omnicast import ARIMAForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    >>> model = ARIMAForecaster(order=(1, 0, 0)).fit(y)
    >>> model.predict(horizon=2).mean.round(2).tolist()
    [14.76, 15.55]

    Pass ``seasonal_order=(P, D, Q, m)`` for SARIMA, and an ``X`` DataFrame
    (with matching index at both ``fit`` and ``predict`` time) to include
    exogenous regressors.

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - You already know (or want to fix) the ARIMA/SARIMA order, or
           need exogenous regressors -- the only model in this package
           that supports them
       * - Avoid when
         - You want the order searched automatically (see
           :class:`AutoARIMAForecaster`)
       * - Handles trend
         - Yes, via differencing (``d``) and/or ``trend``
       * - Handles seasonality
         - Yes, via ``seasonal_order``
       * - Extra dependencies
         - None (uses ``statsmodels``)
       * - Min. observations
         - Enough for ``SARIMAX`` to estimate the requested order;
           roughly ``sum(order) + sum(seasonal_order[:3]) + 1`` at minimum
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str | None = None,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend

    def fit(self, y, X=None):
        from ..utils import validate_series

        self.y_ = validate_series(y)
        model = SARIMAX(
            self.y_,
            exog=X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result_ = model.fit(disp=False)
        self.params_ = self.result_.params.copy()
        self.parameter_confidence_intervals_ = self.result_.conf_int(alpha=0.05)
        self.aic_ = float(self.result_.aic)
        self.bic_ = float(self.result_.bic)
        self.fitted_values_ = pd.Series(
            self.result_.fittedvalues, index=self.y_.index, name="fitted"
        )
        self.residuals_ = (self.y_ - self.fitted_values_).rename("residual")
        self.sigma2_ = float(np.nanmean(np.square(self.residuals_)))
        self.n_obs_, self.is_fitted_ = len(self.y_), True
        self._future_exog_ = None
        return self

    def predict(self, horizon, X=None, level=(80, 95)):
        self._future_exog_ = X
        return super().predict(horizon, X=None, level=level)

    def _fit(self, y: pd.Series) -> None:  # handled by fit
        pass

    def _fitted_values(self) -> np.ndarray:
        return np.asarray(self.result_.fittedvalues)

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.result_.get_forecast(
            horizon, exog=getattr(self, "_future_exog_", None)
        )
        return np.asarray(prediction.predicted_mean), np.asarray(prediction.se_mean)


class AutoARIMAForecaster(ARIMAForecaster):
    """Select a non-seasonal or seasonal ARIMA by corrected AIC grid search.

    Examples
    --------
    >>> import pandas as pd
    >>> from omnicast import AutoARIMAForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    >>> model = AutoARIMAForecaster(max_p=1, max_d=1, max_q=0).fit(y)
    >>> model.order_
    (0, 1, 0)
    >>> model.predict(horizon=2).mean.round(2).tolist()
    [14.0, 14.0]

    Pass ``seasonal_period`` to also search ``(P, D, Q, m)``.

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - You want ARIMA/SARIMA without hand-picking an order; the
           default first candidate :class:`~omnicast.AutoForecaster`
           tries
       * - Avoid when
         - Interactive use with a wide search space -- cost scales as the
           product of every ``max_*`` bound plus one, and it is often the
           slowest model in the package to fit
       * - Handles trend
         - Yes, searched via ``max_d``
       * - Handles seasonality
         - Yes, searched via ``seasonal_period`` + ``max_P``/``max_D``/``max_Q``
       * - Extra dependencies
         - None (uses ``statsmodels``)
       * - Min. observations
         - Enough for the largest candidate order to be estimated;
           failed candidates are skipped rather than aborting the search
    """

    def __init__(
        self,
        seasonal_period: int | None = None,
        max_p: int = 2,
        max_d: int = 1,
        max_q: int = 2,
        max_P: int = 1,
        max_D: int = 1,
        max_Q: int = 1,
        information_criterion: str = "aicc",
    ):
        self.seasonal_period = seasonal_period
        self.max_p, self.max_d, self.max_q = max_p, max_d, max_q
        self.max_P, self.max_D, self.max_Q = max_P, max_D, max_Q
        self.information_criterion = information_criterion
        self.trend = None

    def fit(self, y, X=None):
        from itertools import product

        from ..utils import validate_series

        series = validate_series(y)
        seasonal_orders = [(0, 0, 0, 0)]
        if self.seasonal_period and self.seasonal_period > 1:
            seasonal_orders = [
                (P, D, Q, self.seasonal_period)
                for P, D, Q in product(
                    range(self.max_P + 1), range(self.max_D + 1), range(self.max_Q + 1)
                )
            ]
        candidates, best = [], None
        for order in product(range(self.max_p + 1), range(self.max_d + 1), range(self.max_q + 1)):
            for seasonal_order in seasonal_orders:
                if sum(order) + sum(seasonal_order[:3]) == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = SARIMAX(
                            series,
                            exog=X,
                            order=order,
                            seasonal_order=seasonal_order,
                            trend=None,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False, maxiter=100)
                    k, n = len(result.params), result.nobs
                    aicc = result.aic + (2 * k * (k + 1) / (n - k - 1)) if n > k + 1 else np.inf
                    score = (
                        aicc
                        if self.information_criterion == "aicc"
                        else getattr(result, self.information_criterion)
                    )
                    candidates.append(
                        {
                            "order": order,
                            "seasonal_order": seasonal_order,
                            "aic": result.aic,
                            "aicc": aicc,
                            "bic": result.bic,
                        }
                    )
                    if best is None or score < best[0]:
                        best = (score, order, seasonal_order)
                except (ValueError, np.linalg.LinAlgError):
                    continue
        if best is None:
            raise RuntimeError("No ARIMA candidate could be fitted")
        self.search_results_ = (
            pd.DataFrame(candidates).sort_values(self.information_criterion).reset_index(drop=True)
        )
        self.order_, self.seasonal_order_ = best[1], best[2]
        self.order, self.seasonal_order = self.order_, self.seasonal_order_
        return super().fit(series, X=X)
