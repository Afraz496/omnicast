"""Optional TimesFM foundation model forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseForecaster

try:
    import timesfm
except ImportError:  # pragma: no cover - exercised when timesfm is not installed
    timesfm = None


class TimesFMForecaster(BaseForecaster):
    """Zero-shot time-series foundation forecaster wrapping Google Research's TimesFM.

    Requires the optional ``timesfm`` extra (``pip install omnicast[timesfm]``).

    TimesFM is a pretrained decoder-only transformer model designed for zero-shot
    time-series forecasting across diverse frequencies and domains without requiring
    per-series gradient training or fine-tuning.

    By default, this forecaster loads the TimesFM 2.5 checkpoint
    (``google/timesfm-2.5-200m-pytorch``). TimesFM 3.0 checkpoints
    (such as ``google/timesfm-3.0-pytorch``) are also supported via ``repo_id``,
    subject to their non-commercial / research license terms.

    Prediction intervals use residual-variance random-walk scaling
    (:math:`\\sqrt{\\sigma^2 \\cdot h}`) derived from empirical in-sample residuals.

    Not included in :class:`~omnicast.AutoForecaster`'s default candidate list:
    foundation models require optional dependencies and external weight loading.
    Pass it explicitly via ``AutoForecaster(models=[...])`` to include it in model selection.

    Parameters
    ----------
    repo_id : str, default "google/timesfm-2.5-200m-pytorch"
        Hugging Face model repository ID or local path to checkpoint.
    context_len : int, default 512
        Maximum context window length to feed the foundation model.
    horizon_len : int, default 128
        Maximum forecast horizon configured for compiled model backends.
    normalize_inputs : bool, default True
        Whether to normalize the input series before inference.
    torch_compile : bool, default False
        Whether to compile the PyTorch model graph for faster inference.
    device : str or None, default None
        Target device for execution (e.g., ``"cpu"``, ``"cuda"``, ``"mps"``).
    model : object or None, default None
        Optional pre-instantiated TimesFM model or compatible mock. When supplied,
        automatic checkpoint downloading is bypassed.
    compute_fitted : bool, default True
        Whether to compute empirical 1-step rolling in-sample fitted values on the
        training tail for residual variance calculation. If False, naive persistence
        is used for baseline residual variance.

    Examples
    --------
    >>> import pandas as pd
    >>> from omnicast import TimesFMForecaster
    >>> y = pd.Series([10.0, 11.5, 12.0, 14.2, 15.0, 16.1])
    >>> # Using an existing model instance or mock:
    >>> # model = TimesFMForecaster().fit(y)
    >>> # forecast = model.predict(horizon=3)

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - Zero-shot forecasting on diverse series without local training.
       * - Avoid when
         - You need an ultra-lightweight environment, or cannot download 
           external model checkpoints
       * - Handles trend
         - Yes, learned representations capture diverse trend structures
       * - Handles seasonality
         - Yes, zero-shot attention over input context length
       * - Extra dependencies
         - ``timesfm``, ``torch`` (``pip install omnicast[timesfm]``)
    """

    def __init__(
        self,
        repo_id: str = "google/timesfm-2.5-200m-pytorch",
        context_len: int = 512,
        horizon_len: int = 128,
        normalize_inputs: bool = True,
        torch_compile: bool = False,
        device: str | None = None,
        model: object | None = None,
        compute_fitted: bool = True,
    ):
        self.repo_id = repo_id
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.normalize_inputs = normalize_inputs
        self.torch_compile = torch_compile
        self.device = device
        self.model = model
        self.compute_fitted = compute_fitted

    def _load_model(self) -> object:
        if "timesfm-3" in self.repo_id:
            if not hasattr(timesfm, "TimesFM3Forecaster"):
                raise ImportError(
                    f"Installed timesfm does not support TimesFM 3.0 for {self.repo_id}"
                )
            return timesfm.TimesFM3Forecaster.from_pretrained(
                self.repo_id,
                device=self.device,
            )

        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise ImportError(
                f"Installed timesfm does not support TimesFM 2.5 for {self.repo_id}"
            )
        tfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.repo_id,
            torch_compile=self.torch_compile,
        )
        tfm_model.compile(
            timesfm.ForecastConfig(
                max_context=self.context_len,
                max_horizon=max(self.horizon_len, 128),
                normalize_inputs=self.normalize_inputs,
            )
        )
        return tfm_model

    def _fit(self, y: pd.Series) -> None:
        if timesfm is None and self.model is None:
            raise ImportError(
                "TimesFMForecaster requires the optional 'timesfm' dependency; "
                "install with `pip install omnicast[timesfm]`."
            )
        n = len(y)
        if n < 3:
            raise ValueError(
                f"TimesFMForecaster requires at least 3 observations, got {n}"
            )

        self._n = n
        self._context = y.to_numpy(dtype=float)

        if self.model is not None:
            self._model = self.model
        else:
            self._model = self._load_model()

        self._fitted_vals = self._compute_fitted_values(self._context)

    def _compute_fitted_values(self, context: np.ndarray) -> np.ndarray:
        fitted = np.full(self._n, np.nan)
        if not self.compute_fitted:
            # Fallback to persistence (1-step naive lag)
            fitted[1:] = context[:-1]
            return fitted

        # Compute 1-step rolling forecasts over the recent tail (up to 16 points)
        start_idx = max(2, self._n - 16)
        for i in range(start_idx, self._n):
            sub_ctx = context[:i]
            try:
                mean_1, _ = self._predict_array(sub_ctx, horizon=1)
                fitted[i] = float(mean_1[0])
            except Exception:  # noqa: BLE001
                # If sub-context inference fails, fall back to persistence
                fitted[i] = context[i - 1]
        return fitted

    def _predict_array(
        self, ctx: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        # Handle models with predict() (TimesFM 3 or unified wrappers)
        if hasattr(self._model, "predict"):
            res = self._model.predict(
                context=ctx,
                horizon=horizon,
                return_quantiles=True,
            )
            if hasattr(res, "forecast"):
                mean = np.asarray(res.forecast, dtype=float)
                quantiles = getattr(res, "quantiles", None)
                return mean, quantiles
            if isinstance(res, tuple):
                return np.asarray(res[0], dtype=float), res[1]
            return np.asarray(res, dtype=float), None

        # Handle models with forecast() (TimesFM 2.5)
        if hasattr(self._model, "forecast"):
            point, quantiles = self._model.forecast(horizon=horizon, inputs=[ctx])
            quant = quantiles[0] if quantiles is not None else None
            return np.asarray(point[0], dtype=float), quant

        # Handle callable mock / function
        if callable(self._model):
            res = self._model(ctx, horizon)
            if isinstance(res, tuple):
                return np.asarray(res[0], dtype=float), res[1]
            return np.asarray(res, dtype=float), None

        raise TypeError(f"Unsupported model backend: {type(self._model).__name__}")

    def _fitted_values(self) -> np.ndarray:
        return self._fitted_vals

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        mean, _ = self._predict_array(self._context, horizon=horizon)
        steps = np.arange(1, horizon + 1)
        base_sigma2 = (
            self.sigma2_
            if getattr(self, "sigma2_", 0.0) > 0
            else float(np.var(self._context)) or 1.0
        )
        se = np.sqrt(base_sigma2 * steps)
        return mean, se
