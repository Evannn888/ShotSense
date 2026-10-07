"""Actual default/optional scene JPEG comparison on frozen development inputs."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.lut_natural import process_jpeg
from artifacts.experiments.fivek_current_effect_v1.evaluate import stats

DATA = ROOT/'data/external/scene_policy_connection_v1'
PRIOR_DATA = ROOT/'data/external/fivek_current_effect_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open('x') as f:
        f.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def sheet(names, records, destination, public=False):
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 20)
    canvas = Image.new('RGB', (1800, 440*len(names)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, identity in enumerate(names):
        stem = Path(identity).stem
        notes = ', '.join(records[identity]['result']['scene_policy']['reasons']) or 'existing limits'
        draw.text((12, 440*row+8), identity+' | '+notes, font=font, fill='black')
        paths = ([DATA/'public_original'/(stem+'.png'), DATA/'public_previous'/(stem+'.png')]
                 if public else [PRIOR_DATA/'original'/(stem+'.png'), PRIOR_DATA/'automatic'/(stem+'.png')])
        paths.append(DATA/'automatic'/(stem+'.png'))
        for col, (path, label) in enumerate(zip(paths, ('Decoded input', 'Previous automatic', 'Scene-aware beta'))):
            with Image.open(path) as im:
                frame = ImageOps.contain(im, (580, 370))
                canvas.paste(frame, (600*col+(600-frame.width)//2, 440*row+38+(370-frame.height)//2))
            draw.text((col*600+12, 440*row+412), label, font=font, fill='black')
    canvas.save(destination, icc_profile=(ROOT/'artifacts/experiments/fivek_paired_targets_v1/srgb.icc').read_bytes())


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    baseline, inputs, implementation = [read(HERE/(n+'.json')) for n in ('baseline', 'input_baseline', 'implementation_baseline')]
    for path, expected in implementation.items():
        assert sha(ROOT/path) == expected, path
    DATA.mkdir(exist_ok=False)
    for folder in ('automatic', 'public_jpeg', 'public_original', 'public_previous', 'comparisons'):
        (DATA/folder).mkdir()
    (HERE/'records').mkdir(exist_ok=False)
    icc = (ROOT/'artifacts/experiments/fivek_paired_targets_v1/srgb.icc').read_bytes()
    records = []
    started = time.perf_counter()
    all_inputs = [(p, False) for p in inputs['photos']] + [(p, True) for p in inputs['public']]
    for index, (item, public) in enumerate(all_inputs, 1):
        identity, stem = item['id'], Path(item['id']).stem
        if public:
            assert sha(item['path']) == item['sha256']
            jpeg = DATA/'public_jpeg'/(stem+'.jpg')
            with Image.open(item['path']) as im:
                im.save(jpeg, quality=100, subsampling=0, icc_profile=icc)
        else:
            jpeg = ROOT/item['jpeg_path']
            assert sha(jpeg) == item['jpeg_sha256']
        old_metadata, old = process_jpeg(jpeg)
        assert old_metadata['processing_version'] == 'natural-adaptive-lut-v2'
        if not public:
            for name, key in (('original', 'original'), ('adjusted', 'previous')):
                path = ROOT/item[key+'_path']
                assert sha(path) == item[name+'_export_sha256']
                with Image.open(path) as im:
                    assert np.array_equal(np.array(im), old[name])
        result, pixels = process_jpeg(jpeg, scene_aware=True)
        adjusted = pixels['adjusted']
        assert adjusted.shape == old['original'].shape and adjusted.dtype == np.uint8
        assert np.isfinite(adjusted).all()
        black = int((np.all(adjusted==0, axis=-1) & ~np.all(old['original']==0, axis=-1)).sum())
        full = int((np.any(adjusted==255, axis=-1) & ~np.any(old['original']==255, axis=-1)).sum())
        assert black == full == 0
        hashes = {}
        exports = [('automatic', adjusted)]
        if public:
            exports += [('public_original', pixels['original']), ('public_previous', old['adjusted'])]
        for folder, rgb in exports:
            destination = DATA/folder/(stem+'.png')
            Image.fromarray(rgb).save(destination, icc_profile=icc)
            with Image.open(destination) as im:
                assert np.array_equal(np.array(im), rgb) and im.info['icc_profile'] == icc
            hashes[str(destination.relative_to(ROOT))] = sha(destination)
        if not public:
            assert sha(ROOT/item['target_path']) == item['target_sha256']
            with Image.open(ROOT/item['target_path']) as im:
                target = np.array(im)
        else:
            target = old['original']
        record = {'id': identity, 'public': public, 'synthetic': item.get('synthetic', False),
                  'size': [adjusted.shape[1], adjusted.shape[0]],
                  'original': stats(old['original'], target), 'previous': stats(old['adjusted'], target),
                  'scene': stats(adjusted, target), 'result': result,
                  'new_black_pixel_positions': black, 'new_full_pixel_positions': full,
                  'exports_sha256': hashes, 'jpeg_sha256': sha(jpeg),
                  'default_matches_frozen_previous': not public}
        save(HERE/'records'/(stem+'.json'), record)
        records.append(record)
        if index % 10 == 0:
            print('Processed', index, '/', len(all_inputs), flush=True)
    by_id = {r['id']: r for r in records}
    for page in range(0, 12, 4):
        sheet(baseline['review_ids'][page:page+4], by_id, DATA/'comparisons'/('review-%d.png' % (page//4+1)))
    sheet(['milky_way', 'shadow_portrait', 'sunset_silhouette', 'city_night'], by_id, DATA/'comparisons/public-controls.png', public=True)
    sheet(['a4796-20090208_at_17h53m27__mg_0089.dng', 'a2654-img_0032.dng'], by_id, DATA/'comparisons/summary.png')
    first = inputs['photos'][0]
    _, repeated = process_jpeg(ROOT/first['jpeg_path'], scene_aware=True)
    with Image.open(DATA/'automatic'/(Path(first['id']).stem+'.png')) as im:
        assert np.array_equal(np.array(im), repeated['adjusted'])
    for path, expected in baseline['protected_sha256'].items():
        if path not in baseline['approved_changes']:
            assert sha(ROOT/path) == expected, path
    for path, expected in {**baseline['runtime_scene_assets'], **implementation}.items():
        assert sha(ROOT/path) == expected, path
    fivek = [r for r in records if not r['public']]
    result = {'fivek_count': len(fivek), 'public_originals': 8, 'synthetic_controls': 2,
              'default_previous_exact_count': len(fivek), 'fixed_repeat_exact': True,
              'recognition_available_count': sum(r['result']['scene_analysis']['status']=='available' for r in records),
              'reason_counts': {reason: sum(reason in r['result']['scene_policy']['reasons'] for r in records)
                                for reason in ('night_mood', 'sunset_silhouette', 'uncertain_lighting', 'very_dark_source', 'document_preserved')},
              'previous_reference_mae_mean': float(np.mean([r['previous']['pooled_rgb_reference_mae'] for r in fivek])),
              'scene_reference_mae_mean': float(np.mean([r['scene']['pooled_rgb_reference_mae'] for r in fivek])),
              'lower_reference_error_count': sum(r['scene']['pooled_rgb_reference_mae'] < r['previous']['pooled_rgb_reference_mae'] for r in fivek),
              'higher_reference_error_count': sum(r['scene']['pooled_rgb_reference_mae'] > r['previous']['pooled_rgb_reference_mae'] for r in fivek),
              'zero_new_endpoint_positions_count': len(records), 'elapsed_seconds': time.perf_counter()-started,
              'scope': 'Reused <=960px FiveK/public development views. Public comparison distance is to decoded input, not an expert. No independent preference acceptance, training, sharpening or default promotion.'}
    save(HERE/'results.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
