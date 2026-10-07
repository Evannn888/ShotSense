"""Bounded, offline RawTherapee recipes; separate from Adobe parameter prediction."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image, ImageCms, PngImagePlugin

from src.jpeg_inference import MAX_JPEG_PIXELS, decode_jpeg

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'artifacts/photo_engine'
CLI = ROOT / 'data/tools/rawtherapee-5.11/rawtherapee-cli'
VERSION = 'rawtherapee-recipes-v5'
RECIPES = ('auto', 'neutral', 'natural', 'chroma', 'shadows')
FINE_TUNING_VERSION = 'rawtherapee-fine-tuning-v1'
FINE_TUNING_LIMITS = {'warmth': (-20, 20), 'tint': (-20, 20),
                      'vibrance': (-20, 20), 'local_contrast': (0, 20)}


def validate_fine_tuning(settings):
    if not isinstance(settings, dict) or set(settings) - set(FINE_TUNING_LIMITS):
        raise ValueError('Unknown fine-tuning settings')
    values = {}
    for name, (low, high) in FINE_TUNING_LIMITS.items():
        value = settings.get(name, 0)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or not low <= value <= high or value != int(value)):
            raise ValueError('Invalid fine-tuning value: ' + name)
        values[name] = int(value)
    return values


def fine_tune_photo(rgb, settings):
    started = time.perf_counter(); settings = validate_fine_tuning(settings)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3
            or min(rgb.shape[:2]) < 1 or rgb.shape[0] * rgb.shape[1] > MAX_JPEG_PIXELS):
        raise ValueError('Invalid fine-tuning source pixels')
    rgb = np.ascontiguousarray(rgb)
    metadata = {'version': FINE_TUNING_VERSION, 'settings': settings,
                'input_pixels_sha256': hashlib.sha256(memoryview(rgb)).hexdigest(),
                'size': [rgb.shape[1], rgb.shape[0]], 'engine_applied': any(settings.values()),
                'semantics': 'Relative sRGB edits of the fixed recipe result; not original camera Kelvin, RAW labels, recovered detail or semantic region masks.'}
    if not metadata['engine_applied']:
        adjusted = rgb.copy()
    else:
        identity = engine_identity()
        wb = settings['warmth'] != 0 or settings['tint'] != 0
        profile = ('[Version]\nAppVersion=5.11\nVersion=351\n[White Balance]\nEnabled=' + str(wb).lower() + '\nSetting=Custom\n'
                   f"Temperature={round(6504 * 2 ** (settings['warmth'] / 100))}\n"
                   f"Green={2 ** (-settings['tint'] / 100):.8f}\nEqual=1\n"
                   '[Vibrance]\nEnabled=' + str(settings['vibrance'] != 0).lower() + '\n'
                   f"Pastels={settings['vibrance']}\nSaturated=0\nPSThreshold=0;75;\n"
                   'ProtectSkins=true\nAvoidColorShift=true\nPastSatTog=false\n'
                   '[Local Contrast]\nEnabled=' + str(settings['local_contrast'] != 0).lower() + '\n'
                   f"Radius=80\nAmount={settings['local_contrast'] / 100}\nDarkness=0.5\nLightness=0.5\n")
        with tempfile.TemporaryDirectory(prefix='shotsense-fine-') as directory:
            folder = Path(directory); source = folder / 'input.png'; output = folder / 'output.png'
            Image.fromarray(rgb).save(source, icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
            layer = folder / 'fine.pp3'; layer.write_text(profile)
            env = dict(os.environ, RT_SETTINGS=str(folder / 'settings'), RT_CACHE=str(folder / 'cache'),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            completed = subprocess.run([str(CLI), '-q', '-O', str(output), '-p', str(BUNDLE / 'neutral.pp3'),
                                        '-p', str(layer), '-n', '-b8', '-c', str(source)], env=env,
                                       capture_output=True, text=True, timeout=45)
            if completed.returncode:
                raise ValueError('Fine tuning failed: ' + (completed.stderr or completed.stdout)[-400:])
            with Image.open(output) as image:
                if image.format != 'PNG' or image.mode != 'RGB' or image.size != (rgb.shape[1], rgb.shape[0]):
                    raise ValueError('Fine tuning changed the source dimensions or format')
                adjusted = np.asarray(image, dtype=np.uint8).copy()
        metadata.update(engine=identity, profile_sha256=hashlib.sha256(profile.encode()).hexdigest(),
                        neutral_profile_sha256=hashlib.sha256((BUNDLE / 'neutral.pp3').read_bytes()).hexdigest())
    metadata['output_pixels_sha256'] = hashlib.sha256(memoryview(adjusted)).hexdigest()
    new_full = 0
    for row in range(0, len(rgb), 128):
        new_full += int((np.any(adjusted[row:row+128] == 255, axis=-1)
                        & ~np.any(rgb[row:row+128] == 255, axis=-1)).sum())
    metadata['new_full_channel_fraction'] = new_full / (rgb.shape[0] * rgb.shape[1])
    metadata['seconds'] = time.perf_counter() - started
    return metadata, adjusted


def engine_identity():
    if not CLI.is_file() or not (BUNDLE / 'manifest.json').is_file():
        raise ValueError('Photo engine is missing. Run venv/bin/python scripts/prepare_photo_engine.py.')
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    if manifest['version'] != '5.11' or hashlib.sha256(CLI.read_bytes()).hexdigest() != manifest['cli_sha256']:
        raise ValueError('Photo engine version/checksum mismatch; prepare the verified engine again.')
    return manifest


def select_recipe(rgb):
    # ponytail: fixed scene statistics cannot infer artistic intent; learn user preferences only after collecting choices.
    small = np.asarray(Image.fromarray(rgb).resize((128, 128)), dtype=np.float32) / 255
    linear = np.where(small <= .04045, small / 12.92, ((small + .055) / 1.055) ** 2.4)
    luma = linear @ np.array([.2126, .7152, .0722], dtype=np.float32)
    median, high = [float(v) for v in np.quantile(luma, [.5, .95])]
    selected = 'natural' if .025 <= median < .25 and high < .9 else 'unchanged'
    return selected, {'median_linear_luminance': median, 'p95_linear_luminance': high,
                      'reason': 'moderately dim' if selected == 'natural' else 'preserve ordinary, bright or extreme-dark intent',
                      'semantics': 'Provisional fixed rule, not calibrated aesthetic confidence.'}


def process_photo(path, recipe='auto'):
    started = time.perf_counter(); path = Path(path)
    if recipe not in RECIPES:
        raise ValueError('Unknown photo recipe')
    if path.suffix.lower() not in ('.jpg', '.jpeg', '.dng'):
        raise ValueError('Photo engine currently accepts DNG, JPG and JPEG')
    if not 0 < path.stat().st_size <= 128 * 1024 * 1024:
        raise ValueError('Photo must be between 1 byte and 128 MB')
    identity = engine_identity()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    is_raw = path.suffix.lower() == '.dng'
    timings = []; profile_hashes = {}
    with tempfile.TemporaryDirectory(prefix='shotsense-engine-') as directory:
        folder = Path(directory)
        env = dict(os.environ, RT_SETTINGS=str(folder / 'settings'), RT_CACHE=str(folder / 'cache'),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        if is_raw:
            import rawpy
            with rawpy.RawPy() as header:
                header.open_file(str(path))
                if not 0 < header.sizes.raw_width * header.sizes.raw_height <= 40_000_000:
                    raise ValueError('DNG exceeds the 40 megapixel limit')
                wb = np.asarray(header.camera_whitebalance[:3])
                if not np.isfinite(wb).all() or (wb <= 0).any():
                    raise ValueError('DNG lacks valid camera white balance information')
            source = folder / 'input.dng'; source.write_bytes(path.read_bytes())
            raw_profile = folder / 'raw.pp3'
            raw_profile.write_text('[White Balance]\nEnabled=true\nSetting=Camera\n'
                                   '[Color Management]\nInputProfile=(cameraICC)\n')
            profile_hashes['raw.pp3'] = hashlib.sha256(raw_profile.read_bytes()).hexdigest()
            profile_applied = False
        else:
            original, profile_applied = decode_jpeg(path)
            source = folder / 'input.png'
            Image.fromarray(original).save(source, icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())

        def run(selected):
            profiles = [BUNDLE / 'neutral.pp3']
            if is_raw: profiles.append(raw_profile)
            if selected in ('natural', 'chroma'): profiles.append(BUNDLE / 'natural.pp3')
            if selected == 'chroma': profiles.append(BUNDLE / 'chroma.pp3')
            if selected == 'shadows': profiles.append(BUNDLE / 'shadows.pp3')
            destination = folder / (selected + '.png')
            command = [str(CLI), '-q', '-O', str(destination)]
            for profile in profiles:
                command.extend(['-p', str(profile)])
                profile_hashes[profile.name] = hashlib.sha256(profile.read_bytes()).hexdigest()
            forward = time.perf_counter()
            completed = subprocess.run(command + ['-n', '-b8', '-c', str(source)], env=env,
                                       capture_output=True, text=True, timeout=45)
            timings.append({'recipe': selected, 'seconds': time.perf_counter() - forward})
            if completed.returncode:
                raise ValueError('Photo engine failed: ' + (completed.stderr or completed.stdout)[-400:])
            with Image.open(destination) as image:
                if image.format != 'PNG' or image.mode != 'RGB' or not 0 < image.width * image.height <= MAX_JPEG_PIXELS:
                    raise ValueError('Photo engine returned an unsupported image')
                rgb = np.asarray(image, dtype=np.uint8).copy()
            if not is_raw and rgb.shape != original.shape:
                raise ValueError('Photo engine changed the source dimensions')
            return rgb

        if is_raw: original = run('neutral')
        selected, selection = select_recipe(original) if recipe == 'auto' else (recipe, {'reason': 'explicit recipe'})
        adjusted = original.copy() if selected == 'unchanged' or (is_raw and selected == 'neutral') else run(selected)
        if adjusted.shape != original.shape:
            raise ValueError('Photo engine changed the baseline dimensions')
        clipped = np.any(adjusted == 255, axis=-1) & ~np.any(original == 255, axis=-1)
        selection['candidate_new_full_channel_fraction'] = float(clipped.mean())
        if recipe == 'auto' and clipped.mean() > .001:
            adjusted = original.copy(); selected = 'unchanged'
            selection['reason'] = 'candidate rejected by clipping guard'
    result = {'schema_version': 1, 'input_format': 'DNG' if is_raw else 'JPEG',
              'input_status': 'experimental_photo_engine', 'processing_version': VERSION,
              'input_sha256': source_hash, 'requested_recipe': recipe, 'selected_recipe': selected,
              'selection': selection, 'engine': identity, 'profile_sha256': profile_hashes,
              'output_size': [original.shape[1], original.shape[0]],
              'recommended_absolute': {}, 'experimental_absolute': {},
              'source_semantics': 'Neutral engine RAW development with camera WB' if is_raw else 'EXIF-oriented, color-managed decoded JPEG',
              'jpeg_color': None if is_raw else {'embedded_icc_converted_to_srgb': profile_applied, 'without_icc': 'assumed sRGB'},
              'quality_status': 'experimental; independent personal-photo preference acceptance pending',
              'timing': {'engine_runs': timings, 'end_to_end_seconds': time.perf_counter() - started}}
    return result, {'original': original, 'adjusted': adjusted}


def render_photo(source, strength=1.):
    original, adjusted = source['original'], source['adjusted']
    if (original.dtype != np.uint8 or adjusted.dtype != np.uint8 or original.ndim != 3
            or original.shape[-1] != 3 or adjusted.shape != original.shape
            or not 0 < original.shape[0] * original.shape[1] <= MAX_JPEG_PIXELS
            or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Invalid photo pixels or strength')
    # ponytail: display-sRGB blending is an interaction control, not a calibrated exposure change.
    pixels = original.copy()
    for row in range(0, len(pixels), 128):
        end = row + 128
        pixels[row:end] = np.rint(original[row:end].astype(np.float32) * (1 - strength)
                                  + adjusted[row:end].astype(np.float32) * strength).astype(np.uint8)
    def png(rgb):
        output = io.BytesIO(); info = PngImagePlugin.PngInfo(); info.add(b'sRGB', b'\x00')
        Image.fromarray(rgb).save(output, format='PNG', pnginfo=info)
        return output.getvalue()
    before, after = png(original), png(pixels)
    metadata = {'renderer_version': VERSION, 'strength': float(strength), 'blend_space': 'display sRGB',
                'size': [pixels.shape[1], pixels.shape[0]], 'full_resolution': True,
                'changed_pixel_fraction': float(np.any(pixels != original, axis=-1).mean()),
                'output_png_sha256': hashlib.sha256(after).hexdigest(),
                'semantics': 'Engine recipe blended with decoded JPEG or neutral RAW baseline; not Adobe parameters.'}
    return before, after, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo', type=Path); parser.add_argument('--recipe', choices=RECIPES, default='auto')
    parser.add_argument('--fine-tune', type=json.loads, help='Internal relative sRGB tuning settings; input is a bounded uint8 RGB .npy')
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--preview-source', type=Path, required=True)
    args = parser.parse_args()
    if args.fine_tune is not None:
        if args.photo.suffix != '.npy' or not 0 < args.photo.stat().st_size <= MAX_JPEG_PIXELS * 3 + 65536:
            raise ValueError('Invalid internal fine-tuning input')
        pixels = np.load(args.photo, allow_pickle=False, mmap_mode='r')
        result, adjusted = fine_tune_photo(pixels, args.fine_tune)
        source = {'adjusted': adjusted}
    else:
        result, source = process_photo(args.photo, args.recipe)
    np.savez_compressed(args.preview_source, **source)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
