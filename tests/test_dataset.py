import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from src.dataset import ParameterNormalizer, ShotSenseDataset, load_or_create_splits, create_dataloaders
from src.parameters import PARAM_MIN, PARAM_MAX


def cached_dataset(tmp_path, count=20):
    images = tmp_path / 'images'; images.mkdir()
    ids = np.array(['a%04d.dng' % i for i in range(count)])
    for i, image_id in enumerate(ids):
        Image.new('RGB', (224,224), (20+i,50,180)).save(images / (Path(image_id).stem+'.jpg'), quality=95)
    y = np.tile([.5,-2.,5.,15000.,2.,10.], (count,1)).astype(np.float32)
    path = tmp_path / 'metadata.npz'
    np.savez_compressed(path, X=np.arange(count*132,dtype=np.float32).reshape(count,132),Y=y,
                        Y_std=np.zeros_like(y),ids=ids,config_hash=np.array('test-version'))
    return ShotSenseDataset(path)


def test_normalizer_roundtrip_and_boundaries():
    data = np.stack([PARAM_MIN,PARAM_MAX,(PARAM_MIN+PARAM_MAX)/2,[1,-2,5,15000,0,10]])
    np.testing.assert_allclose(ParameterNormalizer.denormalize(ParameterNormalizer.normalize(data)),data)
    np.testing.assert_allclose(ParameterNormalizer.normalize(data)[:3],np.stack([-np.ones(6),np.ones(6),np.zeros(6)]))
    tensor = torch.tensor(data,dtype=torch.float64)
    torch.testing.assert_close(ParameterNormalizer.denormalize(ParameterNormalizer.normalize(tensor)),tensor)
    for bad in (np.full(6,np.nan),PARAM_MAX+1,np.ones(5)):
        with pytest.raises(ValueError): ParameterNormalizer.normalize(bad)
    with pytest.raises(ValueError): ParameterNormalizer.denormalize(np.ones(6)*1.01)


def test_tensor_batch_and_persistent_partition(tmp_path):
    ds = cached_dataset(tmp_path)
    image,physical,target = ds[0]
    assert image.shape == (3,224,224) and physical.shape == (132,) and target.shape == (6,)
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in (image,physical,target))
    np.testing.assert_array_equal(physical.numpy(),ds.X[0])
    splits=load_or_create_splits(ds)
    assert [len(splits[name]) for name in ('train','val','test')]==[16,2,2]
    assert load_or_create_splits(ds)==splits
    batch=next(iter(create_dataloaders(ds,splits,batch_size=4)['train']))
    assert [tuple(t.shape) for t in batch]==[(4,3,224,224),(4,132),(4,6)]
    groups={image_id:'g%d'%(i//2) for i,image_id in enumerate(ds.ids)}
    grouped=load_or_create_splits(ds,tmp_path/'grouped.json',groups=groups)
    membership={image_id:name for name,ids in grouped.items() for image_id in ids}
    assert all(membership[ds.ids[i]]==membership[ds.ids[i+1]] for i in range(0,20,2))
    with pytest.raises(ValueError): load_or_create_splits(ds,data_version='changed')


def test_manifest_and_path_integrity(tmp_path):
    ds=cached_dataset(tmp_path)
    manifest={'status':'complete','scope':'pilot','metadata_sha256':hashlib.sha256(ds.metadata_path.read_bytes()).hexdigest(),'config_hash':'test-version'}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    ds=ShotSenseDataset(ds.metadata_path)
    with pytest.raises(ValueError,match='Pilot'): load_or_create_splits(ds)
    assert load_or_create_splits(ds,pilot=True)
    manifest['metadata_sha256']='corrupt'; (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='SHA256'): ShotSenseDataset(ds.metadata_path)
    (tmp_path/'manifest.json').unlink()
    np.savez(ds.metadata_path,X=ds.X,Y=ds.Y,Y_std=ds.Y_std,ids=np.array(['../evil.dng']+ds.ids[1:].tolist()))
    with pytest.raises(ValueError,match='Unsafe'): ShotSenseDataset(ds.metadata_path)
