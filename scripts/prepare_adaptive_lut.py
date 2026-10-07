"""Prepare the pinned paired sRGB author weights; never called during inference."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'data/external_models/image_adaptive_3dlut'
REVISION = 'b491f6df64a588864739a157db271e5c848e1805'
BASE = 'https://raw.githubusercontent.com/HuiZeng/Image-Adaptive-3DLUT/' + REVISION + '/'
FILES = {
    'pretrained_models/sRGB/classifier.pth': (1083261, '439f2ada316f45d1d28dd86a161790aa3b4e334b'),
    'pretrained_models/sRGB/LUTs.pth': (1294452, '0830dceee2256ea5d0965c029945c329fac82a26'),
    'models.py': (12014, '8fbc2de9c12a474de774880a524198d54fb25fb7'),
    'models_x.py': (8474, 'fb3c44475ec62682a062d84ed89601f6a4458a0e'),
    'demo_eval.py': (2793, 'f6dd73969940e4eb6a21f3f9ef587289bba712df'),
    'trilinear_cpp/src/trilinear.cpp': (7615, 'b4d75b04c36b97559d7421b2ff2df98ee16405a2'),
    'LICENSE': (11357, '261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64'),
    'README.md': (5488, 'eae43b086353d920e83042098dbefad24a642267'),
}


def prepare():
    files = {}
    for name, (size, blob) in FILES.items():
        path = BUNDLE / name
        if path.exists():
            data = path.read_bytes()
        else:
            with urllib.request.urlopen(BASE + name, timeout=30) as response:
                data = response.read(size + 1)
        identity = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != size or identity != blob:
            raise ValueError('Upstream artifact size/blob mismatch: ' + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as file: file.write(data)
        files[name] = {'bytes': size, 'git_blob': blob, 'sha256': hashlib.sha256(data).hexdigest(), 'url': BASE + name}
    manifest = {'author_repository': 'https://github.com/HuiZeng/Image-Adaptive-3DLUT',
                'revision': REVISION, 'variant': 'paired-sRGB', 'repository_license': 'Apache-2.0',
                'files': files, 'semantics': 'Frozen external color/tone model, not validated Adobe parameters or a personal-photo preference model.'}
    text = json.dumps(manifest, indent=2) + '\n'
    destination = BUNDLE / 'manifest.json'
    if destination.exists() and destination.read_text() != text:
        raise ValueError('Existing LUT manifest differs; inspect the bundle before replacing it')
    destination.write_text(text)
    return manifest


if __name__ == '__main__':
    manifest = prepare()
    print('Prepared paired sRGB LUT bundle at ' + str(BUNDLE))
    print('Verified ' + str(len(manifest['files'])) + ' pinned artifacts')
