"""Exploratory scene checks with the deployed model; no retraining or label selection."""
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from src.inference import predict_dng
from src.preview import VERSION, prepare_preview_source, render_linear_preview
from src.preprocess import ROOT, sha256_file

# Chosen by visible scene from a fixed random contact sheet, before inference.
CASES = [('portrait', 'a3503'), ('dark-interior', 'a3953'), ('night-building', 'a3861'),
         ('backlit-ruins', 'a3246'), ('snow-mountain', 'a3954'), ('sunset', 'a2270'),
         ('colorful-textiles', 'a0575'), ('cloudy-lake', 'a2049'),
         ('test-portrait', 'a2132'), ('test-backlit-sky', 'a0715'),
         ('test-night-street', 'a3998'), ('test-mountain-valley', 'a0113')]


def main():
    output = ROOT / 'artifacts/preview_variety'
    output.mkdir(exist_ok=True)
    paths = list((ROOT / 'data/raw/dngs').glob('*.dng'))
    splits = json.loads((ROOT / 'data/processed/splits.json').read_text())['splits']
    font = ImageFont.load_default(size=20)
    entries = []
    for scene, prefix in CASES:
        matches = [p for p in paths if p.name.lower().startswith(prefix + '-')]
        assert len(matches) == 1, (prefix, matches)
        path = matches[0]
        started = time.perf_counter()
        result, _ = predict_dng(path)
        params = {k: result['recommended_absolute'][k] for k in ('Exposure', 'HighlightRecovery')}
        source = prepare_preview_source(path)
        before, full, full_meta = render_linear_preview(source, params)
        _, soft, soft_meta = render_linear_preview(source, params, strength=.75)
        zero_before, zero_after, _ = render_linear_preview(source, params, strength=0)
        assert zero_before == zero_after == before
        variants = [('before', before), ('100pct', full), ('75pct', soft)]
        sheet = Image.new('RGB', (1500, 430), '#f3f4f6')
        draw = ImageDraw.Draw(sheet)
        for i, (name, payload) in enumerate(variants):
            (output / (scene + '-' + name + '.png')).write_bytes(payload)
            image = ImageOps.contain(Image.open(io.BytesIO(payload)).convert('RGB'), (490, 355))
            sheet.paste(image, (i * 500 + (500-image.width)//2, 45+(355-image.height)//2))
            draw.text((i*500+12, 12), name, fill='#202020', font=font)
        draw.text((12, 405), scene + ' | ' + path.name, fill='#202020', font=font)
        sheet.save(output / (scene + '-comparison.jpg'), quality=94)
        entry = {'scene': scene, 'photo_id': path.name,
                 'split': next(k for k, v in splits.items() if path.name.lower() in v),
                 'parameters': params, 'full': full_meta, 'soft': soft_meta,
                 'zero_strength_exact': True, 'elapsed_seconds': time.perf_counter()-started,
                 'comparison': scene+'-comparison.jpg'}
        entries.append(entry)
        print(json.dumps(entry), flush=True)
    for group in range((len(entries)+3)//4):
        sheet = Image.new('RGB', (1500, 1720), 'white')
        for row, entry in enumerate(entries[group*4:group*4+4]):
            sheet.paste(Image.open(output/entry['comparison']), (0,row*430))
        sheet.save(output/('overview-'+str(group+1)+'.jpg'), quality=94)
    report = {'selection': 'Visual scene selection before inference: eight from 80 full-data candidates, four from 40 held-out test candidates; NumPy seed 73. Exploratory only, no labels consulted.',
              'renderer_version': VERSION, 'preview_sha256': sha256_file(ROOT/'src/preview.py'),
              'model_sha256': sha256_file(ROOT/'artifacts/model/model.onnx'), 'photos': entries,
              'limits': 'Approximate global rendering, no Lightroom reference; no claim of RAW highlight recovery or population quality.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
