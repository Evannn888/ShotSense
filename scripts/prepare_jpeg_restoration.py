"""Explicitly download the pinned author checkpoint, verify it, and install atomically."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'artifacts/jpeg_restoration'
manifest = json.loads((folder / 'manifest.json').read_text())
weights = folder / 'model.safetensors'
if weights.exists() and hashlib.sha256(weights.read_bytes()).hexdigest() == manifest['weights_sha256']:
    print('Pinned JPEG enhancement weights are already prepared.')
else:
    url = manifest['weights_url'] + '/resolve/' + manifest['weights_revision'] + '/model.safetensors'
    data = urllib.request.urlopen(url, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != manifest['weights_sha256']:
        raise ValueError('Downloaded checkpoint checksum mismatch')
    temporary = weights.with_suffix('.download')
    temporary.write_bytes(data)
    temporary.replace(weights)
    print('Pinned JPEG enhancement weights prepared locally.')
