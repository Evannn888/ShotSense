"""Bounded Phase D validation-only head search on the accepted stretch input."""
import json
from pathlib import Path
import shutil

import numpy as np

from src.model import ROOT
from src.preprocess import sha256_file
import src.train as training

BASE = ROOT / 'artifacts/experiments/model_vnext'
CONTROL = BASE / 'control-seed42'
GRID = [('H1', .001, .01), ('H2', .001, .001), ('H3', .0003, .01), ('H4', .0003, .001)]


def check_run(path, seed, lr, decay):
    config = json.loads((path / 'config.json').read_text())
    reference = json.loads((CONTROL / 'config.json').read_text())
    expected = dict(reference, seed=seed, learning_rate=lr, weight_decay=decay)
    if config != expected:
        raise ValueError('Run configuration mismatch: ' + str(path))
    for name, value in config['implementation_sha256'].items():
        if sha256_file(ROOT / 'src' / name) != value:
            raise ValueError('Changed trainer implementation: ' + name)
    for name, value in json.loads((path / 'frozen_run.json').read_text())['files'].items():
        if sha256_file(path / name) != value:
            raise ValueError('Changed frozen artifact: ' + name)
    if json.loads((path / 'run_status.json').read_text())['status'] != 'complete':
        raise ValueError('Run is incomplete')
    if (path / 'final_test_access.json').exists():
        raise ValueError('Run has accessed test data')
    return json.loads((path / 'evaluation.json').read_text())


def run(row, seed):
    name, lr, decay = row
    path = CONTROL if name == 'H1' and seed == 42 else BASE / f'head-{name.lower()}-seed{seed}'
    if not path.exists():
        training.train(ROOT / 'data/processed/metadata.npz', path, seed=seed,
                       split_path=ROOT / 'data/processed/model_vnext/splits.json',
                       groups_path=ROOT / 'data/processed/model_vnext/groups.json',
                       learning_rate=lr, weight_decay=decay)
    return path, check_run(path, seed, lr, decay)


def main():
    check_run(CONTROL, 42, .001, .01)
    original = training.semantic_features

    def reuse(model, dataset, cache_path, batch_size=32):
        cache_path = Path(cache_path)
        if not cache_path.exists():
            shutil.copyfile(CONTROL / 'semantic_features.npz', cache_path)
        return original(model, dataset, cache_path, batch_size)

    training.semantic_features = reuse
    screened = [(row, *run(row, 42)) for row in GRID]
    winner = min(screened, key=lambda item: item[2]['validation']['dual']['p2'])[0]
    pairs = [(seed, run(GRID[0], seed), run(winner, seed)) for seed in (42, 43, 44)]
    reference = json.loads((CONTROL / 'split_manifest.json').read_text())['splits']['val']
    with np.load(ROOT / 'data/processed/metadata.npz', allow_pickle=False) as data:
        index = {v: i for i, v in enumerate(data['ids'].tolist())}
        target = data['Y'][[index[v] for v in reference]][:, [0, 5]] / np.array([8., 100.])
    errors = []
    for seed, control, candidate in pairs:
        values = []
        for path, report in (control, candidate):
            with np.load(path / 'validation_predictions.npz', allow_pickle=False) as data:
                if data['ids'].tolist() != reference:
                    raise ValueError('Validation ordering mismatch')
                # Convert normalized predictions to original units before range scaling.
                prediction = (data['dual'][:, [0, 5]].astype(np.float64) + 1) / 2
                prediction[:, 0] -= .5
                values.append(np.abs(prediction - target).mean(axis=1))
        errors.append(values)
    for losses, (_, control, candidate) in zip(errors, pairs):
        for values, (_, report) in zip(losses, (control, candidate)):
            if not np.isclose(values.mean(), report['validation']['dual']['p2'], atol=1e-8, rtol=1e-6):
                raise ValueError('Bootstrap score differs from the trainer metric')
    mean_errors = np.asarray(errors).mean(axis=0)
    groups = json.loads((ROOT / 'data/processed/model_vnext/groups.json').read_text())
    group_names = sorted({groups[v] for v in reference})
    members = [np.array([i for i, v in enumerate(reference) if groups[v] == name]) for name in group_names]
    sums = np.array([mean_errors[:, ids].sum(axis=1) for ids in members]); counts = np.array([len(ids) for ids in members])
    rng = np.random.default_rng(20261002)
    sampled = rng.integers(0, len(members), (10000, len(members)))
    scores = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)[:, None]
    interval = np.quantile(1 - scores[:, 1] / scores[:, 0], [.025, .975]).tolist()
    control_p2 = [c[1]['validation']['dual']['p2'] for _, c, _ in pairs]
    candidate_p2 = [c[1]['validation']['dual']['p2'] for _, _, c in pairs]
    improvement = 1 - np.mean(candidate_p2) / np.mean(control_p2)
    field_changes = {}
    for idx, name in ((0, 'Exposure'), (5, 'HighlightRecovery')):
        field_changes[name] = float(np.mean([c[1]['validation']['dual']['mae'][idx] for _, _, c in pairs]) / np.mean([c[1]['validation']['dual']['mae'][idx] for _, c, _ in pairs]) - 1)
    gates = {'improved_two_of_three': sum(b < a for a, b in zip(control_p2, candidate_p2)) >= 2,
             'mean_improvement_at_least_5_percent': bool(improvement >= .05),
             'delivered_fields_within_2_percent': max(field_changes.values()) <= .02,
             'positive_bootstrap_interval': interval[0] > 0}
    report = {'phase': 'D', 'scope': 'validation_only', 'selected_screening_row': winner[0],
              'screening': [{'row': row[0], 'run': str(path.relative_to(ROOT)), 'validation': result['validation']} for row, path, result in screened],
              'paired_runs': [{'seed': seed, 'control': str(c[0].relative_to(ROOT)), 'candidate': str(n[0].relative_to(ROOT)), 'control_validation': c[1]['validation'], 'candidate_validation': n[1]['validation']} for seed, c, n in pairs],
              'control_p2_mean': float(np.mean(control_p2)), 'control_p2_std': float(np.std(control_p2, ddof=1)),
              'candidate_p2_mean': float(np.mean(candidate_p2)), 'candidate_p2_std': float(np.std(candidate_p2, ddof=1)),
              'relative_mean_improvement': float(improvement), 'field_relative_changes': field_changes,
              'bootstrap': {'method': 'Paired validation-photo-group bootstrap after averaging photo losses across seeds; photo-weighted', 'seed': 20261002, 'resamples': 10000, 'relative_improvement_95_percent_interval': interval},
              'research_gates': gates, 'decision': 'eligible_for_further_acceptance' if all(gates.values()) else 'retain_control',
              'test_accessed': False, 'production_replaced': False,
              'limitations': ['Validation selected the configuration; bootstrap does not correct selection bias.', 'Grouping remains incomplete and FiveK data were previously inspected.', 'Deployment parity, visual acceptance and latency are not assessed by this head search.']}
    output = BASE / 'phase_d_summary.json'
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: report[key] for key in ('selected_screening_row', 'relative_mean_improvement', 'research_gates', 'decision')}, indent=2))


if __name__ == '__main__':
    main()
