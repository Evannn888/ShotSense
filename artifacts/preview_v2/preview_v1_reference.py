"""Approximate display preview, separate from model inputs and RAW development."""
import io

import numpy as np
from PIL import Image


def render_preview(jpeg, parameters):
    """Always render from the original JPEG; this is not a Lightroom renderer.

    EV scales decoded display RGB; legacy recovery is represented by a custom
    luminance soft knee. Clipped source details cannot be reconstructed.
    """
    if set(parameters) - {'Exposure', 'HighlightRecovery'}:
        raise ValueError('Preview only supports Exposure and HighlightRecovery')
    exposure = float(parameters.get('Exposure', 0))
    recovery = float(parameters.get('HighlightRecovery', 0))
    if not np.isfinite([exposure, recovery]).all() or not -4 <= exposure <= 4 or not 0 <= recovery <= 100:
        raise ValueError('Preview parameters are non-finite or out of range')
    with Image.open(io.BytesIO(jpeg)) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    linear *= 2 ** exposure
    luminance = linear @ np.array([.2126, .7152, .0722], dtype=np.float32)
    highlights = np.maximum(luminance - .6, 0)
    compressed = np.minimum(luminance, .6) + highlights / (1 + (recovery / 100) * highlights / .4)
    ratio = np.divide(compressed, luminance, out=np.ones_like(luminance), where=luminance > 0)
    linear = np.clip(linear * ratio[..., None], 0, 1)
    srgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
    output = np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(output).save(buffer, format='PNG')
    return buffer.getvalue()
