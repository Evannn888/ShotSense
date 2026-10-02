"""Shared, absolute legacy Camera Raw parameter contract."""
import numpy as np

PARAMS = ("Exposure", "Contrast", "Saturation", "Temperature", "Tint", "HighlightRecovery")
PARAM_MIN = np.array([-4, -50, -100, 2000, -150, 0], dtype=np.float64)
PARAM_MAX = np.array([4, 100, 100, 50000, 150, 100], dtype=np.float64)


def validate_parameters(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != len(PARAMS):
        raise ValueError("Parameters must have a final dimension of 6")
    if not np.isfinite(values).all():
        raise ValueError("Parameters contain missing or non-finite values")
    if ((values < PARAM_MIN) | (values > PARAM_MAX)).any():
        raise ValueError("Parameters exceed the declared absolute encoding ranges")
    return values
