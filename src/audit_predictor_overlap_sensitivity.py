"""Post hoc test-subset scoring after excluding exact 42-field train matches.

No model is fitted or tuned. The saved predictions and official raw CSV row order
are used to produce a descriptive sensitivity result.
"""
import collections
import csv
import gzip
import json
import math
from pathlib import Path

from audit_feature_overlap_logloss import RAW, RES, EXCLUDED, CLASSES, hash_values, normalized_label


def metrics(conf, loss_sum, n):
    support = [sum(row) for row in conf]
    predicted = [sum(conf[i][j] for i in range(10)) for j in range(10)]
    tp = [conf[i][i] for i in range(10)]
    f1 = [2 * tp[i] / (support[i] + predicted[i]) if support[i] + predicted[i] else 0 for i in range(10)]
    recall = [tp[i] / support[i] if support[i] else 0 for i in range(10)]
    correct = sum(tp)
    numerator = correct * n - sum(support[i] * predicted[i] for i in range(10))
    denominator = math.sqrt((n * n - sum(x * x for x in support)) * (n * n - sum(x * x for x in predicted)))
    return {'rows': n, 'Macro_F1': sum(f1) / 10,
            'Weighted_F1': sum(f1[i] * support[i] for i in range(10)) / n,
            'Balanced_Accuracy': sum(recall) / 10,
            'MCC': numerator / denominator, 'Accuracy': correct / n,
            'Log_Loss': loss_sum / n,
            'support': dict(zip(CLASSES, support))}


def run():
    train = RAW / 'UNSW_NB15_training-set.csv'
    test = RAW / 'UNSW_NB15_testing-set.csv'
    with train.open(newline='', encoding='utf-8-sig') as stream:
        names = next(csv.reader(stream))
    features = [c for c in names if c not in EXCLUDED]
    train_keys = set()
    with train.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            train_keys.add(hash_values(tuple(row[c] for c in features)))
    shared = []
    labels = []
    with test.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            shared.append(hash_values(tuple(row[c] for c in features)) in train_keys)
            labels.append(normalized_label(row))
    confs = {'official_test': [[0] * 10 for _ in range(10)],
             'no_exact_42_feature_train_match': [[0] * 10 for _ in range(10)],
             'matched_feature_subset': [[0] * 10 for _ in range(10)]}
    losses = collections.Counter()
    counts = collections.Counter()
    with gzip.open(RES / 'test_predictions_and_probabilities.csv.gz', 'rt', newline='', encoding='utf-8') as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            assert i == int(row['row_index']) and row['true_label'] == labels[i]
            y = CLASSES.index(labels[i]); pred = CLASSES.index(row['predicted_label'])
            probs = [float(row[f'prob_{c}']) for c in CLASSES]
            p_true = probs[y] / sum(probs)
            loss = -math.log(max(p_true, 2.220446049250313e-16))
            groups = ['official_test', 'matched_feature_subset' if shared[i] else 'no_exact_42_feature_train_match']
            for group in groups:
                confs[group][y][pred] += 1
                losses[group] += loss
                counts[group] += 1
    assert counts['official_test'] == len(shared) == 82332
    result = {
        'scope': 'Post hoc descriptive scoring using already-saved 82,332 test predictions; no retraining, threshold selection, or feature selection.',
        'matching_rule': 'Exact literal equality across all 42 model input fields, excluding id, attack_cat, and label.',
        'groups': {group: metrics(confs[group], losses[group], counts[group]) for group in confs},
        'removed_test_rows': counts['matched_feature_subset'],
        'remaining_test_rows': counts['no_exact_42_feature_train_match'],
    }
    (RES / 'predictor_overlap_sensitivity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
