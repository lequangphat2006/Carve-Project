"""Unit test cho class weight (Muc III.7 cua CARVE v5.1)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight


def _make_imbalanced_data(n_pos=16, n_neg=224, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_pos + n_neg, n_features)
    y = np.array([1] * n_pos + [0] * n_neg)
    return X, y


def _expected_ratio(n_pos, n_neg):
    return n_neg / n_pos


def test_compute_class_weight_formula():
    n_pos, n_neg = 16, 224
    _, y = _make_imbalanced_data(n_pos, n_neg)
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
    ratio = weights[1] / weights[0]
    expected = _expected_ratio(n_pos, n_neg)
    assert np.isclose(ratio, expected, rtol=0.01)


def test_logreg_accepts_balanced():
    n_pos, n_neg = 16, 224
    X, y = _make_imbalanced_data(n_pos, n_neg)
    clf = LogisticRegression(class_weight='balanced', max_iter=200)
    clf.fit(X, y)
    assert clf.class_weight == 'balanced'


def test_svm_accepts_balanced():
    n_pos, n_neg = 16, 224
    X, y = _make_imbalanced_data(n_pos, n_neg)
    clf = SVC(class_weight='balanced', kernel='linear')
    clf.fit(X, y)
    assert clf.class_weight == 'balanced'


def test_rf_accepts_balanced_subsample():
    n_pos, n_neg = 16, 224
    X, y = _make_imbalanced_data(n_pos, n_neg)
    clf = RandomForestClassifier(
        n_estimators=2, class_weight='balanced_subsample',
        random_state=0, bootstrap=True
    )
    clf.fit(X, y)
    assert clf.class_weight == 'balanced_subsample'
    preds = clf.predict(X)
    assert preds.shape == (n_pos + n_neg,)


def test_bce_with_logits_pos_weight():
    n_pos, n_neg = 16, 224
    expected = _expected_ratio(n_pos, n_neg)
    pos_weight = torch.tensor([expected], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    assert np.isclose(pos_weight.item(), expected, rtol=0.01)
    logits = torch.tensor([[0.0], [0.0]])
    targets = torch.tensor([[1.0], [0.0]])
    loss = loss_fn(logits, targets)
    loss_unweighted = nn.BCEWithLogitsLoss()(logits, targets)
    assert loss.item() > loss_unweighted.item()
