"""Check bounded public listing pages, never fetch dataset archives or weights."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

FOLDER = Path(__file__).resolve().parent


def links(name):
    source = (FOLDER / 'sources' / (name + '.md')).read_text()
    return list(dict.fromkeys(re.findall(r'https?://[^\s<>"\)]+', source)))


selected = []
for key, selector in (
    ('adaptive_lut_fivek', lambda u: 'drive.google.com/drive/folders' in u),
    ('adaptive_lut_fivek', lambda u: 'data.csail.mit.edu' in u),
    ('msec', lambda u: 'drive.google.com/file' in u),
    ('msec', lambda u: 'ln2.sync.com' in u),
    ('ppr10k', lambda u: 'drive.google.com/drive/folders' in u),
    ('lcdp', lambda u: 'drive.google.com/drive/folders' in u),
    ('rendered_wb', lambda u: 'drive.google.com' in u),
):
    selected.append((key, next(u for u in links(key) if selector(u))))
lol_links = [u for u in links('lol_v2') if 'drive.google.com/file' in u]
selected.append(('lol_v2', lol_links[-1]))


def check(item):
    key, url = item
    request = urllib.request.Request(url, headers={'User-Agent': 'ShotSenseDatasetResearch/1.0'})
    result = dict(dataset=key, published_url=url, scope='Public listing only; not archive accessibility/integrity.')
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result.update(http_status=response.status, final_url=response.url,
                          content_type=response.headers.get('Content-Type'),
                          declared_response_length=response.headers.get('Content-Length'))
            data = response.read(256 * 1024)
        if 'text' in (result['content_type'] or ''):
            content = data.decode('utf-8', errors='replace')
            title = re.search(r'<title[^>]*>(.*?)</title>', content, re.I | re.S)
            result['page_title'] = html.unescape(title.group(1)) if title else None
            result['filename_hints'] = sorted(set(re.findall(
                r'[A-Za-z0-9_ .-]{2,80}\.(?:zip|7z|tar\.gz|tif)', content)))[:20]
        result['sampled_response_bytes'] = len(data)
        result['sampled_response_sha256'] = hashlib.sha256(data).hexdigest()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        result.update(http_status=getattr(error, 'code', None), error=str(error))
    return result


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(check, selected))
    (FOLDER / 'access_checks.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    for row in rows:
        print(json.dumps({k: row.get(k) for k in ('dataset', 'http_status', 'page_title', 'filename_hints', 'error')}))
