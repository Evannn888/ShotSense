"""Fixed, previously inspected photo diagnostics; no independent quality claim."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from src.photo_engine import engine_identity, process_photo, render_photo
from src.preview import render_linear_preview


def compare(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('Comparison exists; use a new output directory')
    photos = [('LOL-' + stem, ROOT / 'data/external/lol_pilot' / (stem + '.jpg'))
              for stem in ('27', '102', '113', '254', '565', '6')]
    reviewed = json.loads((ROOT / 'artifacts/preview_v3/report.json').read_text())
    historical_test = {stem.lower() for stem in reviewed['unseen_test_ids']}
    ids = sorted({row['photo_id'] for row in reviewed['photos'] if row['photo_id'].lower() not in historical_test})[:6]
    available = {p.name.lower(): p for p in (ROOT / 'data/raw/dngs').glob('*.dng')}
    photos += [(stem, available[stem.lower()]) for stem in ids]
    fixtures = ROOT / 'data/tools/rawtherapee-5.11/fixtures'; fixtures.mkdir(exist_ok=True)
    for name, rgb in [('daylight', np.full((192, 256, 3), [190, 180, 160], dtype=np.uint8)),
                      ('extreme-dark', np.full((192, 256, 3), [8, 6, 4], dtype=np.uint8)),
                      ('moderately-dim', np.full((192, 256, 3), [65, 80, 95], dtype=np.uint8))]:
        path = fixtures / (name + '.jpg'); Image.fromarray(rgb).save(path, quality=95)
        photos.append(('synthetic-' + name, path))
    manifest = {'scope': 'Previously inspected public regressions and synthetic controls; no external-photo acceptance',
                'engine': engine_identity(), 'recipes': ['neutral', 'natural', 'chroma', 'auto'],
                'photos': [{'id': name, 'path': str(path.relative_to(ROOT)),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in photos]}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    records = []; rows = []; crops = []
    for name, path in photos:
        started = time.perf_counter(); variants = {}; measures = {}; before = None
        for recipe in manifest['recipes']:
            result, source = process_photo(path, recipe)
            before_png, after_png, metadata = render_photo(source)
            if before is None: before = Image.open(io.BytesIO(before_png)).copy()
            variants[recipe] = Image.open(io.BytesIO(after_png)).copy()
            measures[recipe] = {'selected_recipe': result['selected_recipe'], 'selection': result['selection'],
                               'timing': result['timing'], 'export': metadata,
                               'mean_absolute_srgb_change': float(np.abs(source['adjusted'].astype(np.float32)
                                                                         - source['original']).mean() / 255)}
            if recipe == 'neutral':
                original = source['original'].astype(np.float32) / 255
                linear = np.where(original <= .04045, original / 12.92, ((original + .055) / 1.055) ** 2.4)
                _, current, _ = render_linear_preview_small(linear)
                variants['current'] = Image.open(io.BytesIO(current)).copy()
            if recipe == 'auto':
                assert render_photo(source, 0)[0] == render_photo(source, 0)[1]
                (output / (name + '-auto.png')).write_bytes(after_png)
                (output / (name + '-auto.json')).write_text(json.dumps(dict(result, export=metadata), indent=2) + '\n')
        ordered = [('Original / neutral RAW', before), ('Current +0.35EV', variants['current']),
                   ('Engine neutral', variants['neutral']), ('Engine +0.35EV', variants['natural']),
                   ('Engine + chroma', variants['chroma']), ('Automatic proposal', variants['auto'])]
        if name=='a2132-IMG_4947.dng':
            portrait_details(output,before,variants['auto'],variants['chroma'])
        for canvas_rows, crop in ((rows, False), (crops, True)):
            row = Image.new('RGB', (300 * len(ordered), 234), 'white'); draw = ImageDraw.Draw(row)
            for col, (label, image) in enumerate(ordered):
                if crop and col == 1:
                    draw.rectangle((300,34,599,233), fill='#eeeeee')
                    draw.text((305,3), name[:35], fill='black')
                    draw.text((305,18), 'Current preview: no native crop', fill='black')
                    draw.text((305,85), 'Limited to 1600 px; omitted', fill='black')
                    continue
                image = image.copy()
                if crop:
                    w, h = image.size; image = image.crop((max(0,w//2-150),max(0,h//2-100),min(w,w//2+150),min(h,h//2+100)))
                else: image.thumbnail((300, 200))
                row.paste(image, (col * 300, 34)); draw.text((col * 300 + 5, 3), name[:35], fill='black')
                draw.text((col * 300 + 5, 18), label, fill='black')
            canvas_rows.append(row)
        records.append({'id': name, 'size': list(before.size), 'seconds': time.perf_counter() - started,
                        'recipes': measures})
        print(name, before.size, round(records[-1]['seconds'], 2), flush=True)
    for name, panels in [('overview', rows), ('native-crops', crops)]:
        for start in range(0, len(panels), 5):
            sheet = Image.new('RGB', (1800, 234 * len(panels[start:start+5])), 'white')
            for offset, row in enumerate(panels[start:start+5]): sheet.paste(row, (0, 234 * offset))
            sheet.save(output / (name + '-' + str(start//5+1) + '.jpg'), quality=95)
    (output / 'report.json').write_text(json.dumps({'manifest': manifest, 'cases': records}, indent=2) + '\n')


def render_linear_preview_small(linear):
    import cv2
    h, w = linear.shape[:2]; factor = min(1, 1600 / max(h, w))
    linear = np.clip(cv2.resize(linear, (max(1, round(w * factor)), max(1, round(h * factor))), interpolation=cv2.INTER_AREA), 0, 1)
    return render_linear_preview(np.ascontiguousarray(linear, dtype=np.float32), {'Exposure': .35, 'HighlightRecovery': 0})


def portrait_details(output, before, automatic, chroma):
    boxes=[(1120,2420,1450,2590),(1665,2300,1995,2470)]
    sheet=Image.new('RGB',(990,414),'white'); draw=ImageDraw.Draw(sheet)
    for row,box in enumerate(boxes):
        for col,(label,image) in enumerate([('Neutral RAW',before),('Automatic gentle lift',automatic),('Gentle lift + chroma',chroma)]):
            draw.text((col*330+5,row*207+3),label+' / eye '+str(row+1),fill='black')
            sheet.paste(image.crop(box),(col*330,row*207+30))
    sheet.save(output/'portrait-details.jpg',quality=95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/experiments/photo_engine_v1/diagnostics-v2')
    compare(parser.parse_args().output_dir)
