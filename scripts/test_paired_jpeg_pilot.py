"""Training-only LOL data/renderer diagnostic; no learned enhancement model."""
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.jpeg_inference import predict_jpeg
from src.preprocess import ROOT, sha256_file
from src.preview import apply_tone, encode_png, VERSION


def metrics(rgb, target, original):
    mse = float(np.mean((rgb.astype(np.float64) / 255 - target / 255.) ** 2))
    return {'mse': mse, 'psnr_db': float(-10 * np.log10(max(mse, 1e-15))),
            'mean_rgb': float(rgb.mean() / 255),
            'new_full_channel_fraction': float((np.any(rgb == 255, axis=-1)
                                                & ~np.any(original == 255, axis=-1)).mean())}


def main():
    data = ROOT / 'data/external/lol_pilot'
    out = ROOT / 'artifacts/experiments/jpeg_pilot'
    report_path = out / 'report.json'
    if report_path.exists():
        raise FileExistsError('Preserve completed pilot; use a new version for another experiment')
    acquired = json.loads((out / 'acquisition.json').read_text())
    archive = data / acquired['filename']
    assert sha256_file(archive) == acquired['sha256']
    started = time.perf_counter()
    entries = []
    font = ImageFont.load_default(size=16)
    grid = [(e, h) for e in (-2., -1., 0., .5, 1., 1.5, 2., 2.5, 3., 3.5, 4.)
            for h in (0., 50., 100.)]
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive paths'
        members = {}
        for branch in ('low', 'high'):
            members[branch] = {Path(n).name: n for n in names
                               if n.startswith('our485/' + branch + '/')
                               and n.lower().endswith('.png')}
            assert len(members[branch]) == 485, 'Unexpected author training partition'
        assert set(members['low']) == set(members['high'])
        selected = sorted(np.random.default_rng(20261003).choice(
            sorted(members['low']), 24, replace=False).tolist())
        for index, name in enumerate(selected):
            files = {}
            audits = {}
            for branch in ('low', 'high'):
                member = members[branch][name]
                payload = zipped.read(member)  # Explicit member only; never extract eval15.
                path = data / branch / name
                path.parent.mkdir(exist_ok=True)
                if path.exists() and path.read_bytes() != payload:
                    raise ValueError('Existing source differs from the author archive')
                path.write_bytes(payload)
                with Image.open(io.BytesIO(payload)) as image:
                    assert image.format == 'PNG' and image.mode == 'RGB'
                    assert not image.info.get('icc_profile'), 'ICC conversion needs a separate protocol'
                    files[branch] = np.array(image)
                    audits[branch] = {'member': member, 'path': str(path.relative_to(ROOT)),
                                     'sha256': sha256_file(path), 'bytes': len(payload),
                                     'size': list(image.size), 'mode': image.mode,
                                     'png_bit_depth': int(payload[24]), 'icc': 'absent; assumed sRGB'}
            low, target = files['low'], files['high']
            assert low.shape == target.shape and low.dtype == target.dtype == np.uint8
            assert audits['low']['png_bit_depth'] == audits['high']['png_bit_depth'] == 8
            jpeg = data / (Path(name).stem + '.jpg')
            Image.fromarray(low).save(jpeg, quality=95, subsampling=0)
            prediction, source = predict_jpeg(jpeg)
            _, identity = encode_png(source)
            assert source.shape == low.shape, 'Pilot must not change pair geometry'
            with Image.open(jpeg) as decoded:
                assert np.array_equal(identity, np.array(decoded.convert('RGB')))
            zero, _ = apply_tone(source, {'Exposure': 1., 'HighlightRecovery': 50.}, strength=0)
            assert np.array_equal(encode_png(zero)[1], identity)
            params = {k: prediction['experimental_absolute'][k]
                      for k in ('Exposure', 'HighlightRecovery')}
            adjusted, effective = apply_tone(source, params)
            experimental = encode_png(adjusted)[1]
            assert np.array_equal(experimental, encode_png(apply_tone(source, params)[0])[1])
            manual = encode_png(apply_tone(source, {'Exposure': 1., 'HighlightRecovery': 0.})[0])[1]
            best = None
            for exposure, recovery in grid:
                trial = encode_png(apply_tone(source, {'Exposure': exposure,
                                                      'HighlightRecovery': recovery})[0])[1]
                score = metrics(trial, target, identity)
                if best is None or score['mse'] < best[0]['mse']:
                    best = (score, trial, exposure, recovery)
            variants = [('Identity JPEG', identity), ('Experimental estimates', experimental),
                        ('Fixed +1 EV', manual), ('Target-informed grid', best[1]),
                        ('Normal-light target', target)]
            entry = {'id': name, 'partition': 'our485 training; development pilot', 'source': audits,
                     'jpeg_sha256': sha256_file(jpeg), 'jpeg_quality': 95, 'jpeg_subsampling': 0,
                     'jpeg_vs_png': metrics(identity, low, low), 'experimental_parameters': effective,
                     'prediction_timing': prediction['timing'],
                     'target_mean_rgb': float(target.mean() / 255),
                     'metrics': {label: metrics(rgb, target, identity) for label, rgb in variants[:-1]},
                     'grid_best_parameters': {'Exposure': best[2], 'HighlightRecovery': best[3]}}
            entries.append(entry)
            if index < 8:  # Chosen by input index before examining any scores.
                sheet = Image.new('RGB', (1500, 270), 'white')
                draw = ImageDraw.Draw(sheet)
                for col, (label, rgb) in enumerate(variants):
                    image = ImageOps.contain(Image.fromarray(rgb), (290, 200))
                    sheet.paste(image, (col * 300 + (300 - image.width) // 2, 35))
                    draw.text((col * 300 + 6, 8), label, font=font, fill='black')
                draw.text((8, 244), name + ' | training-only development pair', font=font, fill='black')
                sheet.save(out / (Path(name).stem + '-comparison.jpg'), quality=94)
            print(name, 'identity / experimental / fixed / diagnostic PSNR:',
                  [round(entry['metrics'][label]['psnr_db'], 2) for label, _ in variants[:-1]], flush=True)
    summary = {}
    for label in entries[0]['metrics']:
        summary[label] = {metric: float(np.mean([e['metrics'][label][metric] for e in entries]))
                          for metric in entries[0]['metrics'][label]}
        summary[label]['better_than_identity_count'] = sum(
            e['metrics'][label]['mse'] < e['metrics']['Identity JPEG']['mse'] for e in entries)
    report = {'schema_version': 1, 'selection_seed': 20261003, 'count': len(entries),
              'selection': '24 matching training filenames sampled without pixels/targets; eval15 not extracted',
              'source_archive_sha256': acquired['sha256'], 'renderer': VERSION,
              'model_sha256': sha256_file(ROOT / 'artifacts/model/model.onnx'),
              'implementation_sha256': {p: sha256_file(ROOT / p) for p in
                                        ('scripts/test_paired_jpeg_pilot.py', 'src/jpeg_inference.py', 'src/preview.py')},
              'grid': {'exposures': sorted(set(e for e, _ in grid)), 'recoveries': [0, 50, 100],
                       'target_informed': True, 'protect_highlights': True},
              'checks': {'pairs_valid': True, 'zero_strength_identity': True, 'repeatable_render': True,
                         'eval15_not_extracted': True, 'trained_model': False},
              'summary': summary, 'photos': entries, 'elapsed_seconds': time.perf_counter() - started,
              'limits': 'Training-only low-light diagnostic; not held-out quality, general retouching or learned JPEG enhancement acceptance. Target-informed grid is not deployable; absent ICC assumed sRGB.'}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    sheet = Image.new('RGB', (1500, 270 * 8), 'white')
    for i, entry in enumerate(entries[:8]):
        with Image.open(out / (Path(entry['id']).stem + '-comparison.jpg')) as row:
            sheet.paste(row, (0, 270 * i))
    sheet.save(out / 'overview.jpg', quality=94)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
