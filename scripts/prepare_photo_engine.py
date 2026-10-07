"""Install the checksummed official Mac engine without replacing another build."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'data/tools/rawtherapee-5.11'
URL = 'https://rawtherapee.com/shared/builds/mac/RawTherapee_macOS_12.3_Universal_5.11.zip'
ARCHIVE_SHA = 'a3e928cafc7a00a090dcf50b84cd45af48ce8f457269d601d301072df7e0f7af'
CLI_SHA = '7d1e2d68a89f7f425f618acd4b515dce623326b8579237e560885ba9c58aa928'


def tree_hash(folder):
    digest = hashlib.sha256()
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            digest.update(str(path.relative_to(folder)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def prepare():
    FOLDER.mkdir(parents=True, exist_ok=True)
    archive = FOLDER / Path(URL).name
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        with urllib.request.urlopen(URL, timeout=45) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError('RawTherapee archive checksum mismatch')
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(FOLDER / 'unpacked')
    unpacked = FOLDER / 'unpacked/RawTherapee_macOS_12.3_Universal_5.11_folder'
    cli = FOLDER / 'rawtherapee-cli'
    if hashlib.sha256((unpacked / 'rawtherapee-cli').read_bytes()).hexdigest() != CLI_SHA:
        raise ValueError('RawTherapee CLI checksum mismatch')
    mount = FOLDER / 'mount'; mount.mkdir(exist_ok=True)
    attached = subprocess.run(['hdiutil', 'attach', '-readonly', '-nobrowse', '-mountpoint', str(mount),
                               str(next(unpacked.glob('*.dmg')))], input='Y\n', text=True,
                              capture_output=True, timeout=45)
    if attached.returncode:
        raise ValueError('Unable to mount the verified RawTherapee installer: ' + attached.stderr[-300:])
    target = Path('/Applications/RawTherapee.app')
    try:
        expected_app = tree_hash(mount / 'RawTherapee.app')
        if target.exists():
            if tree_hash(target) != expected_app:
                raise ValueError('Refusing to replace a different RawTherapee installation')
        else:
            subprocess.run(['ditto', str(mount / 'RawTherapee.app'), str(target)], check=True, timeout=60)
        if tree_hash(target) != expected_app:
            raise ValueError('Installed RawTherapee application differs from the verified archive')
    finally:
        subprocess.run(['hdiutil', 'detach', str(mount)], check=True, capture_output=True, timeout=30)
    shutil.copyfile(unpacked / 'rawtherapee-cli', cli); cli.chmod(0o755)
    manifest = ROOT / 'artifacts/photo_engine/manifest.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'engine': 'RawTherapee', 'version': '5.11', 'pp3_version': 351,
        'archive_url': URL, 'archive_sha256': ARCHIVE_SHA, 'cli_sha256': CLI_SHA,
        'application_sha256': expected_app, 'platform': 'macOS universal, minimum 12.3',
        'license': 'GPL-3.0-or-later; separate unmodified application',
        'source': 'https://github.com/RawTherapee/RawTherapee/tree/5.11'}, indent=2) + '\n')
    print('Verified RawTherapee 5.11 prepared; photo processing remains offline.')


if __name__ == '__main__':
    prepare()
