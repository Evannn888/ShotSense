import json

import pytest
from scripts.prepare_data import finalize_support,UNSUPPORTED_WB
from src.preprocess import sha256_file


def test_only_confirmed_unsupported_wb_is_excluded(tmp_path,monkeypatch):
    output=tmp_path/'processed'; output.mkdir()
    raw=tmp_path/'raw'; raw.mkdir(); (raw/'bad.dng').write_bytes(b'raw')
    metadata=output/'metadata.npz'; metadata.write_bytes(b'cached valid inputs')
    report={'input_count':2,'success_count':1,'filtered_count':0,'failed_count':1,'filtered_images':{},
            'failed_images':{'bad.dng':UNSUPPORTED_WB},'status':'failed','metadata_sha256':sha256_file(metadata)}
    manifest=output/'manifest.json'; manifest.write_text(json.dumps(report))
    class Header:
        camera_whitebalance=[0.,1.,0.,0.]
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def open_file(self,path): pass
    monkeypatch.setattr('scripts.prepare_data.rawpy.RawPy',Header)
    monkeypatch.setattr('scripts.prepare_data.catalog_context',lambda _:({}, {}, {}))
    final=finalize_support(output,raw)
    assert final['status']=='complete' and final['filtered_count']==1 and final['failed_count']==0
    audit=json.loads((output/'input_support_audit.json').read_text())
    assert audit['unsupported_count']==1 and (output/audit['original_report']).exists()
    assert metadata.read_bytes()==b'cached valid inputs'
    report['failed_images']['bad.dng']='OSError: disk write failed'
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Unresolved'):
        finalize_support(output,raw)
    report['failed_images']['bad.dng']=UNSUPPORTED_WB; manifest.write_text(json.dumps(report))
    Header.camera_whitebalance=[2.,1.,1.5,0.]
    with pytest.raises(ValueError,match='now has valid'):
        finalize_support(output,raw)
