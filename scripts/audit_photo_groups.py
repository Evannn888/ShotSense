"""Propose related-photo pairs from unedited inputs; never merge without review."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.dataset import ShotSenseDataset
from src.extract_labels import open_catalog
from src.preprocess import ROOT, atomic_bytes, sha256_file


def image_hashes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    difference = small[:, 1:] > small[:, :-1]
    spectrum = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].ravel()
    perceptual = spectrum > np.median(spectrum[1:])
    perceptual[0] = False
    return np.packbits(difference.ravel()), np.packbits(perceptual)


def candidate_pairs(difference, perceptual, capture, dhash_limit=4, phash_limit=8):
    # ponytail: bounded O(N^2) hash screening for FiveK; index hashes if the corpus grows substantially.
    bits = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    pairs = {}
    for i in range(len(difference)-1):
        d = bits[difference[i] ^ difference[i+1:]].sum(axis=1)
        p = bits[perceptual[i] ^ perceptual[i+1:]].sum(axis=1)
        for offset in np.flatnonzero((d <= dhash_limit) & (p <= phash_limit)):
            j = i+1+int(offset)
            pairs[i, j] = {'dhash_distance': int(d[offset]), 'phash_distance': int(p[offset]), 'reasons': ['visual_hash']}
    by_camera = defaultdict(list)
    for i, context in enumerate(capture):
        if context is not None:
            camera, serial, timestamp = context
            by_camera[camera, serial].append((timestamp, i))
    for photos in by_camera.values():
        photos.sort()
        for position, (timestamp, i) in enumerate(photos):
            for other_time, j in photos[position+1:]:
                gap = (other_time-timestamp).total_seconds()
                if gap > 3: break
                key = tuple(sorted((i, j)))
                pair = pairs.setdefault(key, {'dhash_distance': int(bits[difference[i]^difference[j]].sum()),
                                              'phash_distance': int(bits[perceptual[i]^perceptual[j]].sum()), 'reasons': []})
                pair['reasons'].append('same_camera_capture_within_3_seconds')
    return pairs


def audit(metadata_path, output_dir):
    dataset = ShotSenseDataset(metadata_path)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    records = defaultdict(set)
    with open_catalog(ROOT/'data/raw/fivek_dataset/raw_photos/fivek.lrcat') as connection:
        for row in connection.execute('''
            SELECT file.lc_idx_filename, model.value, exif.cameraSNRef, image.captureTime
            FROM Adobe_images image JOIN AgLibraryFile file ON file.id_local=image.rootFile
            JOIN AgHarvestedExifMetadata exif ON exif.image=image.id_local
            JOIN AgInternedExifCameraModel model ON model.id_local=exif.cameraModelRef
        '''):
            if row[3]: records[row[0].lower()].add((row[1].strip(), row[2], row[3]))
    capture=[]; differences=[]; perceptual=[]; image_hash=[]
    for photo in dataset.ids:
        values=records.get(str(photo),set()); context=None
        if len(values)==1:
            camera, serial, timestamp=next(iter(values))
            try: context=(camera,serial,datetime.fromisoformat(timestamp))
            except ValueError: pass
        capture.append(context)
        path=dataset.images_dir/(Path(photo).stem+'.jpg')
        image=cv2.imread(str(path))
        if image is None: raise ValueError('Unreadable cached image: '+str(photo))
        d,p=image_hashes(image); differences.append(d); perceptual.append(p); image_hash.append(sha256_file(path))
    candidates=candidate_pairs(np.array(differences),np.array(perceptual),capture)
    pairs=[]
    for (i,j), scores in sorted(candidates.items()):
        pairs.append(dict(scores, left=str(dataset.ids[i]), right=str(dataset.ids[j]), decision='unreviewed', note=''))
    report={'schema_version':1,'metadata_sha256':dataset.data_hash,'source_image_hashes':dict(zip(dataset.ids.tolist(),image_hash)),
            'method':{'dhash_limit':4,'phash_limit':8,'capture_gap_seconds':3,
                      'inputs':'Unedited cached previews and source camera/capture metadata; no expert settings or model errors.',
                      'limitation':'Conservative candidate screening is incomplete; timestamps without camera serial can match different cameras.'},
            'photo_count':len(dataset),'unambiguous_capture_count':sum(c is not None for c in capture),
            'candidate_count':len(pairs),'pairs':pairs,'status':'pending_visual_review'}
    atomic_bytes(output_dir/'candidates.json',json.dumps(report,indent=2).encode())
    for start in range(0,len(pairs),12):
        selected=pairs[start:start+12]
        sheet=Image.new('RGB',(640,len(selected)*180),'white'); draw=ImageDraw.Draw(sheet)
        for row,pair in enumerate(selected):
            for column,key in enumerate(('left','right')):
                with Image.open(dataset.images_dir/(Path(pair[key]).stem+'.jpg')) as source:
                    image=source.convert('RGB'); image.thumbnail((300,150)); sheet.paste(image,(column*320,row*180))
                draw.text((column*320,row*180+152),'%d %s' % (start+row,pair[key]),fill='black')
        sheet.save(output_dir/('review-%03d.jpg' % (start//12)),quality=90)
    print(json.dumps({k:report[k] for k in ('photo_count','unambiguous_capture_count','candidate_count','status')}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,default=ROOT/'data/processed/metadata.npz')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); audit(args.metadata,args.output_dir)
