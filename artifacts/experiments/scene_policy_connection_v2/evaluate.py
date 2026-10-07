"""Reuse the frozen real-worker comparison; compare v2 with immutable v1 outputs."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from artifacts.experiments.scene_policy_connection_v1 import evaluate as probe

probe.HERE = HERE
probe.DATA = ROOT/'data/external/scene_policy_connection_v2'
probe.main()
baseline = probe.read(HERE/'baseline.json')
for path, expected in baseline['previous_controls_sha256'].items():
    assert probe.sha(ROOT/path) == expected, path
records = {r['id']: r for r in (probe.read(p) for p in (HERE/'records').glob('*.json'))}
previous_folder = ROOT/'artifacts/experiments/scene_policy_connection_v1'
prior = {r['id']: r for r in (probe.read(p) for p in (previous_folder/'records').glob('*.json'))}
changed = []
for identity, result in records.items():
    earlier = prior[identity]
    assert earlier['jpeg_sha256'] == result['jpeg_sha256']
    assert earlier['result']['scene_analysis']['views'] == result['result']['scene_analysis']['views']
    old_path = ROOT/'data/external/scene_policy_connection_v1/automatic'/(Path(identity).stem+'.png')
    assert probe.sha(old_path) == earlier['exports_sha256'][str(old_path.relative_to(ROOT))]
    new_path = probe.DATA/'automatic'/(Path(identity).stem+'.png')
    with Image.open(old_path) as old, Image.open(new_path) as new:
        if not np.array_equal(np.array(old), np.array(new)):
            changed.append(identity)
fivek = [key for key, value in records.items() if not value['public']]
summary = {'count': len(records), 'same_classifier_scores_count': len(records),
           'changed_output_count': len(changed), 'changed_ids': changed,
           'v1_fivek_reference_mae_mean': float(np.mean([prior[key]['scene']['pooled_rgb_reference_mae'] for key in fivek])),
           'v2_fivek_reference_mae_mean': float(np.mean([records[key]['scene']['pooled_rgb_reference_mae'] for key in fivek])),
           'lower_reference_error_vs_v1_count': sum(records[key]['scene']['pooled_rgb_reference_mae'] < prior[key]['scene']['pooled_rgb_reference_mae'] for key in fivek),
           'higher_reference_error_vs_v1_count': sum(records[key]['scene']['pooled_rgb_reference_mae'] > prior[key]['scene']['pooled_rgb_reference_mae'] for key in fivek),
           'v1_night_limit_count': sum('night_mood' in r['result']['scene_policy']['reasons'] for r in prior.values()),
           'v2_night_limit_count': sum('night_mood' in r['result']['scene_policy']['reasons'] for r in records.values()),
           'scope': 'Failure-driven development revision on reused inputs, not independent preference or recognition accuracy.'}
probe.save(HERE/'v1_comparison.json', summary)
# Additional v1/v2 pages use the matched decoded original and immutable saved v1 PNG.
font = probe.ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 20)
for page in range(0, 12, 4):
    names = baseline['review_ids'][page:page+4]
    canvas = Image.new('RGB', (1800, 440*len(names)), 'white')
    draw = probe.ImageDraw.Draw(canvas)
    for row, name in enumerate(names):
        stem = Path(name).stem
        draw.text((12, row*440+8), name, font=font, fill='black')
        paths = [ROOT/'data/external/fivek_current_effect_v1/original'/(stem+'.png'),
                 ROOT/'data/external/scene_policy_connection_v1/automatic'/(stem+'.png'),
                 probe.DATA/'automatic'/(stem+'.png')]
        for col, (path, label) in enumerate(zip(paths, ('Decoded input', 'Scene beta v1', 'Scene beta v2'))):
            with Image.open(path) as im:
                frame = probe.ImageOps.contain(im, (580,370))
                canvas.paste(frame, (600*col+(600-frame.width)//2, row*440+38+(370-frame.height)//2))
            draw.text((600*col+12, row*440+412), label, font=font, fill='black')
    canvas.save(probe.DATA/'comparisons'/('v1-v2-%d.png' % (page//4+1)), icc_profile=(ROOT/'artifacts/experiments/fivek_paired_targets_v1/srgb.icc').read_bytes())
print(json.dumps(summary, indent=2), flush=True)
