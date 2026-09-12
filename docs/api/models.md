# Models

## Baselines

```{eval-rst}
.. autoclass:: omnicast.NaiveForecaster
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omnicast.SeasonalNaiveForecaster
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omnicast.MeanForecaster
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omnicast.DriftForecaster
   :members:
   :undoc-members:
   :show-inheritance:
```

## Theta

```{eval-rst}
.. autoclass:: omnicast.ThetaForecaster
   :members:
   :undoc-members:
   :show-inheritance:
```

## Statistical (statsmodels-backed)

```{eval-rst}
.. autoclass:: omnicast.ETSForecaster
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omnicast.ARIMAForecaster
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omnicast.AutoARIMAForecaster
   :members:
   :undoc-members:
   :show-inheritance:
```

## Neural (optional, torch)

```{eval-rst}
.. autoclass:: omnicast.LSTMForecaster
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omnicast.DeepARForecaster
   :members:
   :undoc-members:
   :show-inheritance:
```

## Foundation models (optional, timesfm)

```{eval-rst}
.. autoclass:: omnicast.TimesFMForecaster
   :members:
   :undoc-members:
   :show-inheritance:
```

