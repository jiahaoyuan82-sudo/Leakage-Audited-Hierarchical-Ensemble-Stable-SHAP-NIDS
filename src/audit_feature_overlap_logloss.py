"""Read-only re-audit of saved UNSW-NB15 rows and named probability columns.

Uses only the official CSVs and retained row-level prediction files. It does not fit
or tune a model. Outputs JSON evidence to the staging results directory.
"""
import collections
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get('UNSW_RAW_DATA_DIR', ROOT / 'data' / 'raw')).expanduser().resolve()
RES = ROOT / 'results'
EXCLUDED = {'id', 'label', 'attack_cat'}
CLASSES = ['Normal', 'Analysis', 'Backdoors', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']


def hash_values(values):
    payload = json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).digest()


def normalized_label(row):
    if row['label'] == '0':
        return 'Normal'
    label = row['attack_cat'].strip()
    return {'Backdoor': 'Backdoors', 'Dos': 'DoS'}.get(label, label)


def read_raw(path, feature_names, is_train):
    feature_counts = collections.Counter()
    full_counts = collections.Counter()
    labels = collections.defaultdict(collections.Counter)
    examples = {}
    n = 0
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        assert set(reader.fieldnames) == set(feature_names) | EXCLUDED
        for row in reader:
            n += 1
            features = tuple(row[c] for c in feature_names)
            key = hash_values(features)
            feature_counts[key] += 1
            full_counts[hash_values(tuple(row[c] for c in reader.fieldnames))] += 1
            label = normalized_label(row)
            labels[key][label] += 1
            if key not in examples:
                examples[key] = features
    return {'rows': n, 'feature_counts': feature_counts, 'full_counts': full_counts,
            'labels': labels, 'examples': examples}


def audit_duplicates():
    train_path = RAW / 'UNSW_NB15_training-set.csv'
    test_path = RAW / 'UNSW_NB15_testing-set.csv'
    with train_path.open(newline='', encoding='utf-8-sig') as stream:
        columns = next(csv.reader(stream))
    feature_names = [c for c in columns if c not in EXCLUDED]
    assert len(feature_names) == 42
    train = read_raw(train_path, feature_names, True)
    test = read_raw(test_path, feature_names, False)
    shared = train['feature_counts'].keys() & test['feature_counts'].keys()
    # A digest match is checked against all 42 literal field values.
    assert all(train['examples'][k] == test['examples'][k] for k in shared)
    affected_train = sum(train['feature_counts'][k] for k in shared)
    affected_test = sum(test['feature_counts'][k] for k in shared)
    conflicting_test = sum(
        n for k in shared for label, n in test['labels'][k].items()
        if label not in train['labels'][k]
    )
    same_label_test = affected_test - conflicting_test
    by_test_class = collections.Counter()
    by_test_class_conflict = collections.Counter()
    for k in shared:
        for label, n in test['labels'][k].items():
            by_test_class[label] += n
            if label not in train['labels'][k]:
                by_test_class_conflict[label] += n
    def duplicate_rows(counts):
        return sum(n - 1 for n in counts.values())
    def repeated_rows(counts):
        return sum(n for n in counts.values() if n > 1)
    result = {
        'definition': 'SHA-256 of the 42 model input fields, ordered as in the official CSV; strings compared exactly; digest matches verified against all original field strings',
        'excluded_fields': sorted(EXCLUDED),
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'train_rows': train['rows'], 'test_rows': test['rows'],
        'within_train_full_csv_duplicate_extra_rows': duplicate_rows(train['full_counts']),
        'within_test_full_csv_duplicate_extra_rows': duplicate_rows(test['full_counts']),
        'within_train_feature_duplicate_extra_rows': duplicate_rows(train['feature_counts']),
        'within_test_feature_duplicate_extra_rows': duplicate_rows(test['feature_counts']),
        'within_train_feature_repeated_rows': repeated_rows(train['feature_counts']),
        'within_test_feature_repeated_rows': repeated_rows(test['feature_counts']),
        'cross_split_full_csv_shared_unique_rows': len(train['full_counts'].keys() & test['full_counts'].keys()),
        'cross_split_feature_shared_unique_vectors': len(shared),
        'cross_split_feature_affected_train_rows': affected_train,
        'cross_split_feature_affected_test_rows': affected_test,
        'cross_split_feature_affected_test_fraction': affected_test / test['rows'],
        'cross_split_feature_same_label_test_rows': same_label_test,
        'cross_split_feature_conflicting_label_test_rows': conflicting_test,
        'cross_split_feature_affected_test_rows_by_class': dict(sorted(by_test_class.items())),
        'cross_split_feature_conflicting_test_rows_by_class': dict(sorted(by_test_class_conflict.items())),
    }
    (RES / 'feature_overlap_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def audit_loss(path, prefixes):
    result = {'file': path.name, 'by_model': {}}
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        for prefix, model_name in prefixes:
            assert all(f'{prefix}{c}' in reader.fieldnames for c in CLASSES)
            result['by_model'][model_name] = {
                'negative_log_sum': 0.0, 'rows': 0,
                'max_probability_sum_error': 0.0,
                'min_true_class_probability': 1.0,
                'zero_true_probability_rows': 0,
            }
        for row in reader:
            true = row['true_label']
            assert true in CLASSES
            for prefix, model_name in prefixes:
                vals = [float(row[f'{prefix}{c}']) for c in CLASSES]
                item = result['by_model'][model_name]
                assert all(math.isfinite(v) and v >= 0 for v in vals)
                s = sum(vals)
                item['max_probability_sum_error'] = max(item['max_probability_sum_error'], abs(s - 1))
                p = float(row[f'{prefix}{true}']) / s
                item['min_true_class_probability'] = min(item['min_true_class_probability'], p)
                item['zero_true_probability_rows'] += (p == 0)
                item['negative_log_sum'] += -math.log(max(p, 2.220446049250313e-16))
                item['rows'] += 1
    for item in result['by_model'].values():
        item['mean_log_loss_named_columns'] = item['negative_log_sum'] / item['rows']
        del item['negative_log_sum']
    return result


if __name__ == '__main__':
    overlap = audit_duplicates()
    full = audit_loss(RES / 'test_predictions_and_probabilities.csv.gz',
                      [('prob_', 'full_hierarchical_ensemble'), ('direct_prob_', 'direct_lightgbm')])
    compact = audit_loss(RES / 'compact_test_predictions_and_probabilities.csv.gz',
                         [('prob_', 'compact_hierarchical_ensemble')])
    loss = {'method': 'For each held-out row, take the saved named probability of its true class, normalize by the saved row sum, clip only below machine epsilon, and average -log(p); no model was rerun.',
            'full_prediction_file': full, 'compact_prediction_file': compact}
    (RES / 'probability_logloss_reaudit.json').write_text(json.dumps(loss, indent=2), encoding='utf-8')
    print(json.dumps({'overlap': overlap, 'log_loss': loss}, indent=2))
