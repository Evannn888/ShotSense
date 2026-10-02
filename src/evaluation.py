"""Post-selection diagnostics; Catalog metadata is never a model input."""
from collections import defaultdict
import json

import numpy as np

from src.extract_labels import open_catalog, parse_settings
from src.parameters import PARAMS
from src.preprocess import ROOT


def catalog_context(catalog_path):
    cameras, tags, baseline = defaultdict(set), defaultdict(set), {}
    with open_catalog(catalog_path) as connection:
        for row in connection.execute('''
            SELECT file.lc_idx_filename, model.value
            FROM Adobe_images image JOIN AgLibraryFile file ON file.id_local=image.rootFile
            JOIN AgHarvestedExifMetadata exif ON exif.image=image.id_local
            JOIN AgInternedExifCameraModel model ON model.id_local=exif.cameraModelRef
        '''):
            cameras[row[0].lower()].add(row[1].strip())
        for row in connection.execute('''
            SELECT file.lc_idx_filename, keyword.name
            FROM Adobe_images image JOIN AgLibraryFile file ON file.id_local=image.rootFile
            JOIN AgLibraryKeywordImage ki ON ki.image=image.id_local
            JOIN AgLibraryKeyword keyword ON keyword.id_local=ki.tag
            WHERE keyword.name IS NOT NULL
        '''):
            tags[row[0].lower()].add(row[1])
        records = defaultdict(list)
        for row in connection.execute('''
            SELECT file.lc_idx_filename, develop.text
            FROM AgLibraryCollectionImage collection
            JOIN Adobe_images image ON image.id_local=collection.image
            JOIN AgLibraryFile file ON file.id_local=image.rootFile
            JOIN Adobe_imageDevelopSettings develop ON develop.image=image.id_local
            WHERE collection.collection=943690
        '''):
            if row[1] and row[1].strip(): records[row[0].lower()].append(row[1])
        for photo_id, texts in records.items():
            if len(texts)!=1: continue
            settings=parse_settings(texts[0]); values=[]
            for name in ('Temperature','Tint'):
                key='Custom'+name if settings.get('WhiteBalance')=='Custom' and 'Custom'+name in settings else name
                values.append(settings.get(key))
            if all(isinstance(value,(float,int)) and np.isfinite(value) for value in values):
                if 2000<=values[0]<=50000 and -150<=values[1]<=150:
                    baseline[photo_id]=values
    return {photo:next(iter(values)) if len(values)==1 else 'ambiguous' for photo,values in cameras.items()},tags,baseline


def diagnostics(dataset,predictions,groups,train_ids,metric):
    cameras,tags,baseline=catalog_context(ROOT/'data/raw/fivek_dataset/raw_photos/fivek.lrcat')
    # All thresholds are specified or fitted on train, before reading test errors.
    low,high=np.percentile(dataset.Y[train_ids],[1,99],axis=0)
    # Histogram-bin centers approximate developed Lab lightness, not sensor exposure.
    lightness=dataset.X[:,:32] @ ((np.arange(32)+.5)*100/32)
    dark,bright=np.quantile(lightness[train_ids],[1/3,2/3])
    output={'notes':['Catalog tags overlap and are incomplete; they are not exhaustive scene ground truth.',
                     'Groups smaller than 10 are retained with counts; interpret their errors cautiously.',
                     'As-Shot comparison uses only explicit valid Catalog Temperature/Tint on the same IDs; it is not an online input.'],
            'train_tail_thresholds':{'p1':low.tolist(),'p99':high.tolist()},
            'train_brightness_thresholds':{'lower_tertile':float(dark),'upper_tertile':float(bright),
                'definition':'Lab L histogram-center estimate; thresholds fitted on training inputs only, not sensor exposure.'},'splits':{}}
    for split in groups:
        if split not in ('val','test'): raise ValueError('Diagnostics accept only explicit validation/test partitions')
        indices=groups[split]; buckets=defaultdict(list)
        for index in indices:
            photo=str(dataset.ids[index])
            brightness='dark' if lightness[index]<=dark else 'bright' if lightness[index]>bright else 'middle'
            buckets['brightness/'+brightness].append(int(index))
            buckets['camera/'+cameras.get(photo,'unknown')].append(int(index))
            for tag in tags.get(photo,()): buckets['catalog_tag/'+tag].append(int(index))
            if not tags.get(photo): buckets['catalog_tag/untagged'].append(int(index))
            for column,name in enumerate(PARAMS):
                if dataset.Y[index,column]<low[column] or dataset.Y[index,column]>high[column]:
                    buckets['tail/'+name].append(int(index))
        summaries={name:dict(count=len(ids),**{model:metric(values[ids],dataset.labels.numpy()[ids])
                                              for model,values in predictions.items()})
                   for name,ids in sorted(buckets.items())}
        overlap=np.array([int(i) for i in indices if str(dataset.ids[i]) in baseline],dtype=int)
        wb={'count':len(overlap),'total':len(indices),'coverage':len(overlap)/len(indices)}
        if len(overlap):
            truth=dataset.Y[overlap,3:5].astype(np.float64)
            reference=np.array([baseline[str(dataset.ids[i])] for i in overlap])
            dual=(predictions['dual'][overlap,3:5].astype(np.float64)+1)/2*np.array([48000,300])+np.array([2000,-150])
            wb.update({'as_shot_mae':np.abs(reference-truth).mean(axis=0).tolist(),
                       'dual_mae_same_ids':np.abs(dual-truth).mean(axis=0).tolist(),
                       'as_shot_temperature_mired_mae':float(np.abs(1e6/reference[:,0]-1e6/truth[:,0]).mean()),
                       'dual_temperature_mired_mae_same_ids':float(np.abs(1e6/dual[:,0]-1e6/truth[:,0]).mean())})
        output['splits'][split]={'groups':summaries,'white_balance_comparison':wb}
    return output
