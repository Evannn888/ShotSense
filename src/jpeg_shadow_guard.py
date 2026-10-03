"""Experimental encoded-sRGB gamma guard; bounded gain, no lost-detail recovery."""
import numpy as np

from src.jpeg_curve_candidate import apply_curve


MAX_GAIN = 8.


def guarded_curve(rgb, gamma, strength=1.):
    original = apply_curve(rgb, gamma, 0)  # Reuse the candidate's RGB/gamma contract.
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Invalid curve strength')
    if strength == 0 or np.array_equal(gamma, np.ones(3, dtype=np.float32)):
        return original
    source = original.astype(np.float32) / 255
    limited = np.minimum(source ** np.asarray(gamma, dtype=np.float32), MAX_GAIN * source)
    adjusted = (1-strength)*source + strength*limited
    return np.rint(np.clip(adjusted, 0, 1)*255).astype(np.uint8)
