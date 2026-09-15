"""Optional Chronos foundation model forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseForecaster

try:
    import chronos
except ImportError:  # pragma: no cover - exercised when chronos is not installed
    chronos = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is not installed
    torch = None


class ChronosForecaster(BaseForecaster):
    """Zero-shot time-series foundation forecaster based on Chronos.

    Wraps Amazon Science's Chronos language-model-based forecasting framework.
    Requires the optional ``chronos`` extra (``pip install omnicast[chronos]``).

    Chronos converts continuous time-series values into discrete tokens through
    scaling and quantization, and trains transformer models to predict future
    tokens autoregressively.

    Supports both sample-based models (such as ``amazon/chronos-t5-tiny``)
    and direct quantile models (such as ``amazon/chronos-bolt-tiny``).
    Model weights are distributed under the Apache-2.0 license.

    Not included in :class:`~omnicast.AutoForecaster`'s default candidate list.
    Pass it explicitly via ``AutoForecaster(models=[...])`` to include it.

    Parameters
    ----------
    repo_id : str, default "amazon/chronos-t5-tiny"
        Hugging Face repository ID or local checkpoint path.
    device_map : str, default "cpu"
        Device mapping for model inference (e.g., ``"cpu"``, ``"cuda"``, ``"mps"``).
    torch_dtype : str, default "auto"
        Precision for model parameters (e.g., ``"auto"``, ``"bfloat16"``, ``"float32"``).
    num_samples : int, default 20
        Number of sample paths generated for sample-based models.
    pipeline : object or None, default None
        Optional pre-loaded pipeline instance or mock object. When provided,
        external weight loading is bypassed.
    compute_fitted : bool, default True
        Whether to compute empirical 1-step rolling in-sample fitted values on the
        training tail for residual variance calculation. If False, naive persistence
        is used.

    Examples
    --------
    >>> import pandas as pd
    >>> from omnicast import ChronosForecaster
    >>> y = pd.Series([10.0, 11.2, 12.0, 13.5, 14.1, 15.0])
    >>> # Using a preloaded pipeline or mock:
    >>> # model = ChronosForecaster().fit(y)
    >>> # forecast = model.predict(horizon=2)

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - Zero-shot forecasting across diverse domains without local training
       * - Avoid when
         - You need an ultra-lightweight installation without PyTorch or Transformers,
           or cannot download external model checkpoints
       * - Handles trend
         - Yes, captured by foundation model representations
       * - Handles seasonality
         - Yes, through autoregressive attention over input context
       * - Extra dependencies
         - ``chronos-forecasting``, ``torch`` (``pip install omnicast[chronos]``)
       * - Min. observations
         - 3 observations
    """

    def __init__(
        self,
        repo_id: str = "amazon/chronos-t5-tiny",
        device_map: str = "cpu",
        torch_dtype: str = "auto",
        num_samples: int = 20,
        pipeline: object | None = None,
        compute_fitted: bool = True,
    ):
        self.repo_id = repo_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.num_samples = num_samples
        self.pipeline = pipeline
        self.compute_fitted = compute_fitted

    def _load_pipeline(self) -> object:
        if "bolt" in self.repo_id:
            cls = getattr(chronos, "ChronosBoltPipeline", None)
            if cls is None:
                raise ImportError(
                    f"Installed chronos package does not support ChronosBoltPipeline for {self.repo_id}"
                )
        else:
            cls = getattr(chronos, "ChronosPipeline", None)
            if cls is None:
                raise ImportError(
                    f"Installed chronos package does not support ChronosPipeline for {self.repo_id}"
                )

        kwargs: dict[str, object] = {"device_map": self.device_map}
        if (
            self.torch_dtype != "auto"
            and torch is not None
            and hasattr(torch, self.torch_dtype)
        ):
            kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)

        return cls.from_pretrained(self.repo_id, **kwargs)

    def _fit(self, y: pd.Series) -> None:
        if (chronos is None or torch is None) and self.pipeline is None:
            raise ImportError(
                "ChronosForecaster requires the optional 'chronos' dependency; "
                "install with `pip install omnicast[chronos]`."
            )

        n = len(y)
        if n < 3:
            raise ValueError(
                f"ChronosForecaster requires at least 3 observations, got {n}"
            )

        self._n = n
        self._context = y.to_numpy(dtype=float)

        if self.pipeline is not None:
            self._pipeline = self.pipeline
        else:
            self._pipeline = self._load_pipeline()

        self._fitted_vals = self._compute_fitted_values(self._context)

    def _compute_fitted_values(self, context: np.ndarray) -> np.ndarray:
        fitted = np.full(self._n, np.nan)
        if not self.compute_fitted:
            fitted[1:] = context[:-1]
            return fitted

        start_idx = max(2, self._n - 16)
        for i in range(start_idx, self._n):
            sub_ctx = context[:i]
            try:
                mean_1, _ = self._predict_array(sub_ctx, horizon=1)
                fitted[i] = float(mean_1[0])
            except Exception:  # noqa: BLE001
                fitted[i] = context[i - 1]
        return fitted

    def _predict_array(
        self, ctx: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if hasattr(self._pipeline, "predict"):
            inp = (
                torch.tensor(ctx, dtype=torch.float32)
                if torch is not None
                else ctx
            )
            try:
                out = self._pipeline.predict(
                    inp, prediction_length=horizon, num_samples=self.num_samples
                )
            except TypeError:
                out = self._pipeline.predict(inp, prediction_length=horizon)

            if isinstance(out, list):
                out = out[0]

            if torch is not None and torch.is_tensor(out):
                tensor_data = out.detach().cpu().numpy()
                # Case 1: (batch, num_samples, horizon)
                if tensor_data.ndim == 3 and tensor_data.shape[0] == 1:
                    samples = tensor_data[0]
                    mean = np.mean(samples, axis=0)
                    std = np.std(samples, axis=0)
                    return mean, std
                # Case 2: (num_samples, horizon)
                if tensor_data.ndim == 2:
                    mean = np.mean(tensor_data, axis=0)
                    std = np.std(tensor_data, axis=0)
                    return mean, std
                # Case 3: (horizon,)
                if tensor_data.ndim == 1:
                    return tensor_data, None

            if isinstance(out, np.ndarray):
                if out.ndim == 3:
                    return np.mean(out[0], axis=0), np.std(out[0], axis=0)
                if out.ndim == 2:
                    return np.mean(out, axis=0), np.std(out, axis=0)
                return out, None

        if callable(self._pipeline):
            res = self._pipeline(ctx, horizon)
            if isinstance(res, tuple):
                return np.asarray(res[0], dtype=float), res[1]
            return np.asarray(res, dtype=float), None

        raise TypeError(f"Unsupported pipeline backend: {type(self._pipeline).__name__}")

    def _fitted_values(self) -> np.ndarray:
        return self._fitted_vals

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        mean, se = self._predict_array(self._context, horizon=horizon)
        if se is None:
            steps = np.arange(1, horizon + 1)
            base_sigma2 = (
                self.sigma2_
                if getattr(self, "sigma2_", 0.0) > 0
                else float(np.var(self._context)) or 1.0
            )
            se = np.sqrt(base_sigma2 * steps)
        se = np.maximum(se, 1e-4)
        return mean, se
