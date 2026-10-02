"""Independent RAW preview; custom global tone mapping, not Lightroom."""
import io

import cv2
import numpy as np
from PIL import Image

VERSION = 'linear-raw-shoulder-v3'
MAX_EDGE = 1600


def validate_source(source):
    source = np.asarray(source)
    if (source.dtype != np.float32 or source.ndim != 3 or source.shape[-1] != 3
            or min(source.shape[:2]) < 1 or max(source.shape[:2]) > MAX_EDGE
            or not np.isfinite(source).all() or (source < 0).any() or (source > 1).any()):
        raise ValueError('Preview source must be finite float32 RGB in [0,1], at most 1600 pixels per edge')
    return source


def prepare_preview_source(dng_path):
    # Reuse the verified WB/geometry/linear RAW path; model preprocessing is unchanged.
    from src.color_pipeline import develop_dng_outputs, LIBRAW_SRGB_TO_PROPHOTO
    prophoto, srgb = develop_dng_outputs(dng_path)
    del srgb
    height, width = prophoto.shape[:2]
    factor = min(1, MAX_EDGE / max(height, width))
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    small = cv2.resize(prophoto, size, interpolation=cv2.INTER_AREA)
    linear = small.astype(np.float32) / 65535
    linear = linear @ np.linalg.inv(LIBRAW_SRGB_TO_PROPHOTO).astype(np.float32).T
    # Display-gamut clipping is explicit; no claim of reconstructing clipped RAW.
    return np.ascontiguousarray(np.clip(linear, 0, 1), dtype=np.float32)


def encode_png(linear):
    srgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
    output = np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(output).save(buffer, format='PNG')
    return buffer.getvalue(), output


def apply_tone(source, parameters, strength=1., protect_highlights=True):
    source = validate_source(source)
    if set(parameters) - {'Exposure', 'HighlightRecovery'}:
        raise ValueError('Preview only supports Exposure and HighlightRecovery')
    exposure = float(parameters.get('Exposure', 0))
    recovery = float(parameters.get('HighlightRecovery', 0))
    if not np.isfinite([exposure, recovery, strength]).all() or not -4 <= exposure <= 4 or not 0 <= recovery <= 100 or not 0 <= strength <= 1:
        raise ValueError('Preview parameters are non-finite or out of range')
    effective = {'Exposure': exposure * strength, 'HighlightRecovery': recovery * strength}
    if effective['Exposure']==0 and effective['HighlightRecovery']==0:
        return source.copy(), effective
    gain = 2 ** effective['Exposure']
    peak = source.max(axis=-1)
    if protect_highlights and gain > 1:
        knee = .6 / gain
        delta = np.maximum(peak - knee, 0)
        shoulder = (gain - 1) / (.4 * (1 - knee))
        target = np.where(peak <= knee, gain * peak,
                          .6 + gain * delta / (1 + shoulder * delta))
    else:
        target = gain * peak
    highlights = np.maximum(target - .8, 0)
    target = np.minimum(target, .8) + highlights / (1 + (effective['HighlightRecovery'] / 100) * highlights / .2)
    ratio = np.divide(target, peak, out=np.ones_like(peak), where=peak > 0)
    return np.clip(source * ratio[..., None], 0, 1), effective


def render_linear_preview(source, parameters, strength=1., protect_highlights=True):
    adjusted, effective = apply_tone(source, parameters, strength, protect_highlights)
    before, before_rgb = encode_png(source)
    after, after_rgb = encode_png(adjusted)
    clipped_before = np.any(before_rgb == 255, axis=-1)
    clipped_after = np.any(after_rgb == 255, axis=-1)
    source_ceiling = source >= 1 - 1 / 65535
    metadata = {'renderer_version': VERSION, 'source': 'linear 16-bit RAW ProPhoto converted to display sRGB',
                'size': [int(source.shape[1]), int(source.shape[0])], 'strength': float(strength),
                'protect_highlights': bool(protect_highlights), 'applied_parameters': effective,
                'before_full_channel_fraction': float(clipped_before.mean()),
                'after_full_channel_fraction': float(clipped_after.mean()),
                'new_full_channel_fraction': float((clipped_after & ~clipped_before).mean()),
                'source_channel_ceiling_fraction': float(np.any(source_ceiling, axis=-1).mean()),
                'source_white_ceiling_fraction': float(np.all(source_ceiling, axis=-1).mean()),
                'source_ceiling_semantics': 'Display-linear source ceiling after RAW development, resizing and gamut clipping; not sensor overexposure detection. Dimming cannot reconstruct missing texture.',
                'semantics': 'Custom global exposure/highlight approximation; not Lightroom/PV2003 equivalence; experimental parameters excluded.'}
    return before, after, metadata


def render_preview(jpeg, parameters):
    """Compatibility for previous JPEG preview callers, using the new protection."""
    with Image.open(io.BytesIO(jpeg)) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    return render_linear_preview(linear, parameters)[1]
