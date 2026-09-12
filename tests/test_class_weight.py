"""Unit test cho class weight (Muc III.7 cua CARVE v5.1).

Kiem tra weight_pos/weight_neg xap xi N_neg/N_pos cho moi he thong:
- HandFeat: LogisticRegression, SVM (class_weight='balanced')
- HandFeat: RandomForest (class_weight='balanced_subsample')
- DeepFeat/PretrainedAudio: BCEWithLogitsLoss (pos_weight)
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def _make_imbalanced_data(n_pos=16, n_neg=224, n_features=4, seed=0):
    """Tao du lieu dummy mat can bang lop theo ti le DiCOVA (80/1199 ~ 1:14)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_pos + n_neg, n_features)
    y = np.array([1] * n_pos + [0] * n_neg)
    return X, y


def _expected_ratio(n_pos, n_neg):
    return n_neg / n_pos


def test_logreg_class_weight_balanced():
    """LR voi class_weight='balanced' phai cho weight[1]/weight[0] ~ N_neg/N_pos."""
    n_pos, n_neg = 16, 224
    X, y = _make_imbalanced_data(n_pos, n_neg)
    clf = LogisticRegression(class_weight='balanced', max_iter=200)
    clf.fit(X, y)
    # sklearn luu class_weight_ la array [w_neg, w_pos] theo sorted classes [0, 1]
    actual_ratio = clf.class_weight_[1] / clf.class_weight_[0]
    expected = _expected_ratio(n_pos, n_neg)
    assert np.isclose(actual_ratio, expected, rtol=0.01), (
        f"LR: actual={actual_ratio:.4f}, expected={expected:.4f}"
    )


def test_svm_class_weight_balanced():
    """SVM voi class_weight='balanced' phai cho cung ti le."""
    n_pos, n_neg = 16, 224
    X, y = _make_imbalanced_data(n_pos, n_neg)
    clf = SVC(class_weight='balanced', kernel='linear')
    clf.fit(X, y)
    actual_ratio = clf.class_weight_[1] / clf.class_weight_[0]
    expected = _expected_ratio(n_pos, n_neg)
    assert np.isclose(actual_ratio, expected, rtol=0.01), (
        f"SVM: actual={actual_ratio:.4f}, expected={expected:.4f}"
    )


def test_rf_class_weight_balanced_subsample():
    """RF voi class_weight='balanced_subsample': kiem tra tren 1 cay don.

    balanced_subsample tinh weight tren tung bootstrap sample, khac balanced
    (tinh tren toan bo tap). Test bang cach fit 1 cay va doc class_weight_
    cua cay do; voi bootstrap sample ngau nhien, ti le se xap xi nhung khong
    chinh xac bang balanced. Dung tolerance rong hon (rtol=0.15).
    """
    n_pos, n_neg = 16, 224
    X, y = _make_imbalanced_data(n_pos, n_neg)
    clf = RandomForestClassifier(
        n_estimators=1, class_weight='balanced_subsample',
        random_state=0, bootstrap=True
    )
    clf.fit(X, y)
    # class_weight_ cua RF la aggregated; kiem tra khong NaN va cung dau
    assert clf.class_weight_ is not None
    assert len(clf.class_weight_) == 2
    ratio = clf.class_weight_[1] / clf.class_weight_[0]
    expected = _expected_ratio(n_pos, n_neg)
    # Tolerance rong hon do bootstrap sampling
    assert np.isclose(ratio, expected, rtol=0.15), (
        f"RF: actual={ratio:.4f}, expected={expected:.4f}"
    )


def test_bce_with_logits_pos_weight():
    """BCEWithLogitsLoss: pos_weight duoc ap truoc sigmoid, phai = N_neg/N_pos."""
    n_pos, n_neg = 16, 224
    expected = _expected_ratio(n_pos, n_neg)
    pos_weight = torch.tensor([expected], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    assert np.isclose(pos_weight.item(), expected, rtol=0.01)

    # Sanity check: loss voi pos_weight > 1 phai phat duong tinh manh hon
    logits = torch.tensor([[0.0], [0.0]])
    targets = torch.tensor([[1.0], [0.0]])
    loss = loss_fn(logits, targets)
    loss_unweighted = nn.BCEWithLogitsLoss()(logits, targets)
    assert loss.item() > loss_unweighted.item(), (
        "pos_weight phai lam tang loss tren mau duong"
    )


if __name__ == '__main__':
    test_logreg_class_weight_balanced()
    test_svm_class_weight_balanced()
    test_rf_class_weight_balanced_subsample()
    test_bce_with_logits_pos_weight()
    print("All class_weight tests passed.")