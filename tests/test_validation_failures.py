"""Failure analysis must be complete on validation and exclude test IDs."""
import csv
import json

import numpy as np

from scripts.analyze_validation_failures import analyze
from src.model import ROOT


def test_failure_analysis_matches_validation_and_excludes_test(tmp_path):
    run=ROOT/'artifacts/experiments/model_vnext/control-seed42'
    output=tmp_path/'analysis'
    analyze(run,output)
    split=json.loads((run/'split_manifest.json').read_text())['splits']
    with (output/'validation_residuals.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert {row['photo'] for row in rows}==set(split['val'])
    assert not {row['photo'] for row in rows}&set(split['test'])
    report=json.loads((output/'summary.json').read_text())
    total=sum(bucket['count'] for bucket in report['strata'].values())
    assert total==len(split['val'])
    weighted=sum(bucket['count']*bucket['models']['dual']['p2'] for bucket in report['strata'].values())/total
    reference=json.loads((run/'evaluation.json').read_text())['validation']['dual']['p2']
    assert np.isclose(weighted,reference,atol=1e-8)
