import numpy as np
from src.dataset import ShotSenseDataset
from src.evaluation import diagnostics
from src.preprocess import ROOT
from src.train import metrics


def test_diagnostics_use_same_ids_for_as_shot_comparison(monkeypatch):
    dataset=ShotSenseDataset(ROOT/'data/processed/pilot/metadata.npz')
    first,second=map(str,dataset.ids[:2])
    baseline={first:dataset.Y[0,3:5].tolist()}
    monkeypatch.setattr('src.evaluation.catalog_context',lambda _:({first:'camera-A'}, {first:{'Animal'}}, baseline))
    target=dataset.labels.numpy()
    report=diagnostics(dataset,{'dual':target,'constant':target},
                       {'val':np.array([0,1]),'test':np.array([0,1])},np.arange(2,len(dataset)),metrics)
    wb=report['splits']['val']['white_balance_comparison']
    assert wb['count']==1 and wb['coverage']==.5 and wb['as_shot_mae']==[0,0]
    assert wb['dual_mae_same_ids'][0]<.002
    assert report['splits']['test']['groups']['camera/unknown']['count']==1
    assert report['splits']['test']['groups']['catalog_tag/Animal']['count']==1


def test_brightness_thresholds_fit_train_and_all_baselines_are_reported(monkeypatch):
    dataset=ShotSenseDataset(ROOT/'data/processed/pilot/metadata.npz')
    dataset.X[:,:32]=0
    dataset.X[2:,10]=1
    dataset.X[0,31]=1; dataset.X[1,0]=1
    monkeypatch.setattr('src.evaluation.catalog_context',lambda _:({}, {}, {}))
    target=dataset.labels.numpy()
    predictions={name:target for name in ('constant','physical_linear','semantic','dual')}
    report=diagnostics(dataset,predictions,{'val':np.array([0,1])},np.arange(2,len(dataset)),metrics)
    thresholds=report['train_brightness_thresholds']
    assert thresholds['lower_tertile']==thresholds['upper_tertile']==(10.5*100/32)
    buckets=report['splits']['val']['groups']
    assert buckets['brightness/bright']['count']==buckets['brightness/dark']['count']==1
    assert set(buckets['brightness/bright'])=={'count',*predictions}
    assert 'test' not in report['splits']
