"""Download the predeclared public inputs; prepare views before any inference."""
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
sys.path.insert(0, str(ROOT))
from src.jpeg_inference import decode_jpeg

LOCAL = ROOT / 'data/user_photo_diagnostics/scene-analysis-public-v1'
SELECTED = [
    ('day_portrait', 'Man in denim jacket posing (Unsplash).jpg'),
    ('shadow_portrait', 'Shadowed Portrait (Unsplash).jpg'),
    ('city_night', 'The New York City Skyline at night.jpg'),
    ('milky_way', 'Milky Way 3.jpg'),
    ('sunset_silhouette', 'Joshua tree silhouette at sunset (50330255608).jpg'),
    ('kitchen', 'The Scratch Kitchen Interior.jpg'),
    ('backlit_landscape', 'Crepuscular rays in ggp 2.jpg'),
    ('dog', 'Labrador pup.jpg'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(name, data):
    (FOLDER / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def download(selection):
    key, title = selection
    page = PAGES['File:' + title]
    info = page['imageinfo'][0]
    url = info['url'].split('?')[0]
    original = LOCAL / (key + '.jpg')
    preview = key in PREVIEW_IDS and not original.exists()
    if preview:
        preview_info = PREVIEWS['File:' + title]['imageinfo'][0]
        url = preview_info['thumburl'].split('?')[0]
        original = LOCAL / (key + '.preview.jpg')
    if original.exists():
        data = original.read_bytes()
    else:
        request = urllib.request.Request(url, headers={'User-Agent': 'ShotSenseLocalResearch/1.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(32 * 1024 * 1024 + 1)
    assert 0 < len(data) <= 32 * 1024 * 1024, title
    if not preview:
        assert len(data) == info['size'], title
        assert hashlib.sha1(data).hexdigest() == info['sha1'], title
    assert data.startswith(b'\xff\xd8\xff'), title
    if not original.exists():
        with original.open('xb') as handle:
            handle.write(data)
    rgb, icc = decode_jpeg(original)
    image = Image.fromarray(rgb)
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    path = LOCAL / (key + '.png')
    if not path.exists():
        image.save(path)
    metadata = info['extmetadata']
    def plain(field):
        return html.unescape(re.sub('<[^>]+>', '', metadata.get(field, {}).get('value', '')))
    if preview:
        assert (rgb.shape[1], rgb.shape[0]) == (preview_info['thumbwidth'], preview_info['thumbheight'])
    return dict(id=key, downloaded_path=str(original), path=str(path), title=title,
                source_url=info['descriptionurl'], download_url=url,
                downloaded_sha256=sha(original), thumbnail_sha256=sha(path),
                downloaded_asset_type='commons_960px_preview' if preview else 'commons_original',
                original_sha1_verified=not preview, commons_original_sha1=info['sha1'],
                downloaded_bytes=len(data), commons_original_size=[info['width'], info['height']],
                decoded_size=[int(rgb.shape[1]), int(rgb.shape[0])],
                embedded_icc_converted=icc, author=plain('Artist'),
                license=plain('LicenseShortName'), license_url=plain('LicenseUrl'),
                commons_page_revision=page['revisions'][0])


def contact_sheet(items, path):
    cols = 4 if len(items) > 4 else 2
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * 330, rows * 320), '#f4f4f4')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 18)
    for n, item in enumerate(items):
        x, y = (n % cols) * 330, (n // cols) * 320
        with Image.open(item['path']) as im:
            image = ImageOps.contain(im, (320, 270), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (330 - image.width) // 2, y + (270 - image.height) // 2))
        draw.text((x + 8, y + 276), str(n + 1) + '. ' + item['id'], fill='#111111', font=font)
    canvas.save(path)


if __name__ == '__main__':
    assert not (FOLDER / 'results.json').exists(), 'Preserve completed results; do not overwrite.'
    LOCAL.mkdir(parents=True, exist_ok=True)
    PAGES = {p['title']: p for p in json.loads((FOLDER / 'commons_metadata.json').read_text())['query']['pages'].values()}
    failures_path = FOLDER / 'download_failures_initial.json'
    PREVIEW_IDS = {r['id'] for r in json.loads(failures_path.read_text())} if '--published-previews' in sys.argv else set()
    PREVIEWS = {p['title']: p for p in json.loads((FOLDER / 'commons_preview_metadata.json').read_text())['query']['pages'].values()} if PREVIEW_IDS else {}
    baseline = json.loads((ROOT / 'artifacts/experiments/scene_analysis_research_v1/baseline.json').read_text())
    assert all(sha(ROOT / path) == expected for path, expected in baseline.items())
    save_json('baseline.json', baseline)
    inputs, unavailable = [], []
    for selection in SELECTED:
        if selection[0] == 'sunset_silhouette' and not (LOCAL / 'sunset_silhouette.jpg').exists() and selection[0] not in PREVIEW_IDS:
            unavailable.append(dict(id=selection[0], title=selection[1], http_status=429,
                                    retry_after_seconds=600, reason='Source requested cooldown; no early retry.'))
            continue
        try:
            inputs.append(download(selection))
        except urllib.error.HTTPError as error:
            unavailable.append(dict(id=selection[0], title=selection[1], http_status=error.code,
                                    retry_after_seconds=error.headers.get('Retry-After'), reason=str(error)))
    save_json('unavailable_sources.json', unavailable)
    controls = []
    for key in ('day_portrait', 'backlit_landscape'):
        if not any(item['id'] == key for item in inputs):
            continue
        parent = next(item for item in inputs if item['id'] == key)
        with Image.open(parent['path']) as im:
            srgb = np.asarray(im.convert('RGB'), dtype=np.float64) / 255
        linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4) / 8
        dark = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
        path = LOCAL / (key + '_dark_control.png')
        if not path.exists():
            Image.fromarray(np.rint(np.clip(dark, 0, 1) * 255).astype(np.uint8)).save(path)
        controls.append(dict(id=key + '_dark_control', path=str(path), parent_id=key,
                             parent_thumbnail_sha256=parent['thumbnail_sha256'],
                             thumbnail_sha256=sha(path), synthetic=True,
                             transform='sRGB to linear display RGB, multiply by 1/8, sRGB to rounded uint8'))
    save_json('inputs.json', inputs + controls)
    contact_sheet(inputs, LOCAL / 'inputs_contact_sheet.png')
    control_pairs = []
    for control in controls:
        control_pairs.extend([next(i for i in inputs if i['id'] == control['parent_id']), control])
    contact_sheet(control_pairs, LOCAL / 'controls_contact_sheet.png')
    print(json.dumps({'public_inputs': len(inputs), 'synthetic_controls': len(controls),
                      'unavailable': unavailable, 'folder': str(LOCAL)}))
