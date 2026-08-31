"""Tests for the parts of the pipeline where a silent bug would corrupt a result.

Focus is on the split (leakage / determinism), the metric definitions, the error
analysis arithmetic and Grad-CAM's output contract — not on training quality.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from signvision.config import AugConfig, DataConfig
from signvision.data import make_transforms, select_subset, stratified_split
from signvision.error_analysis import class_error_table, confusion_pairs, error_concentration
from signvision.evaluate import compute_metrics
from signvision.interpretability import GradCAM, activation_maps
from signvision.models import ASLNetOriginal, ASLNetGAP, ASLNetReLU, build_model
from signvision.robustness import default_suite
from signvision.utils import set_seed


@pytest.fixture
def labels() -> np.ndarray:
    return np.repeat(np.arange(6), 50)


def test_split_is_disjoint(labels):
    tr, va, te = stratified_split(labels, DataConfig(images_per_class=None))
    assert set(tr).isdisjoint(va)
    assert set(tr).isdisjoint(te)
    assert set(va).isdisjoint(te)
    assert len(tr) + len(va) + len(te) == len(labels)


def test_split_is_deterministic(labels):
    cfg = DataConfig(images_per_class=None)
    a = stratified_split(labels, cfg)
    b = stratified_split(labels, cfg)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_split_is_stratified(labels):
    """Every class must contribute the *same* number of samples to each partition,
    so no class is over- or under-represented in train, val or test."""
    tr, va, te = stratified_split(labels, DataConfig(images_per_class=None))
    for part, expected in ((tr, 35), (va, 8), (te, 7)):  # 50 per class -> 35/8/7
        counts = np.bincount(labels[part], minlength=6)
        assert set(counts.tolist()) == {expected}, counts


def test_select_subset_respects_images_per_class(labels):
    sel = select_subset(labels, DataConfig(images_per_class=20))
    assert len(sel) == 6 * 20
    assert set(np.bincount(labels[sel]).tolist()) == {20}


def test_select_subset_is_deterministic(labels):
    cfg = DataConfig(images_per_class=20)
    assert np.array_equal(select_subset(labels, cfg), select_subset(labels, cfg))


def test_split_seed_independent_of_train_seed(labels):
    """A different split_seed must change the split; nothing else should."""
    a = stratified_split(labels, DataConfig(images_per_class=None, split_seed=1234))
    b = stratified_split(labels, DataConfig(images_per_class=None, split_seed=99))
    assert not np.array_equal(a[0], b[0])


def test_metrics_include_absent_classes():
    """The bug that made the original 28-image report misleading: a class with no
    support must still appear and must drag the macro average down."""
    classes = list("ABC")
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])
    m = compute_metrics(y, p, classes)
    assert set(m["per_class"]) == set(classes)
    assert m["per_class"]["C"]["support"] == 0
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == pytest.approx(2 / 3)  # C contributes 0
    assert m["weighted_f1"] == pytest.approx(1.0)


def test_metrics_confusion_matrix_shape():
    classes = list("ABCD")
    y = np.random.default_rng(0).integers(0, 4, 40)
    p = np.random.default_rng(1).integers(0, 4, 40)
    m = compute_metrics(y, p, classes)
    cm = np.asarray(m["confusion_matrix"])
    assert cm.shape == (4, 4)
    assert cm.sum() == 40
    assert m["accuracy"] == pytest.approx(np.trace(cm) / 40)


def test_confusion_pairs_and_concentration():
    classes = list("ABC")
    cm = np.array([[8, 2, 0], [3, 7, 0], [1, 0, 9]])
    pairs = confusion_pairs(cm, classes, top_k=5)
    assert (pairs[0]["true"], pairs[0]["predicted"], pairs[0]["count"]) == ("B", "A", 3)
    total = int(cm.sum() - np.trace(cm))
    assert total == 6
    assert sum(p["count"] for p in pairs) == total
    conc = error_concentration(cm, classes)
    assert conc["total_errors"] == 6
    assert conc["top_1_pairs_error_share"] == pytest.approx(3 / 6)
    assert conc["top_10_pairs_error_share"] == pytest.approx(1.0)


def test_class_error_table_sorted_worst_first():
    classes = list("ABC")
    cm = np.array([[10, 0, 0], [5, 5, 0], [0, 0, 10]])
    rows = class_error_table(cm, classes)
    assert rows[0]["class"] == "B"
    assert rows[0]["recall"] == pytest.approx(0.5)
    assert rows[0]["errors"] == 5


def test_eval_transform_is_identity_on_scaling():
    """Evaluation must be a pure uint8 -> [0,1] cast: no augmentation, no normalisation."""
    tf = make_transforms(AugConfig(enabled=True), train=False)
    x = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
    out = tf(x)
    assert out.dtype == torch.float32
    assert torch.allclose(out, x.float() / 255.0)


def test_train_transform_augments_only_when_enabled():
    x = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    off = make_transforms(AugConfig(enabled=False), train=True)
    assert torch.allclose(off(x), x.float() / 255.0)
    set_seed(0)
    on = make_transforms(AugConfig(enabled=True), train=True)
    assert not torch.allclose(on(x), x.float() / 255.0)


def test_no_horizontal_flip_by_default():
    """ASL handshapes are chiral; a mirrored sign is a different (or invalid) sign."""
    assert AugConfig().horizontal_flip is False


@pytest.mark.parametrize("name", ["aslnet_original", "aslnet_relu", "aslnet_gap"])
def test_model_forward_shape_and_params(name):
    m = build_model(name, n_classes=29, image_size=100)
    out = m(torch.randn(2, 3, 100, 100))
    assert out.shape == (2, 29)


def test_relu_variant_differs_only_in_activation():
    """Same parameter count and same conv stack; only the FC non-linearity changes."""
    set_seed(0)
    a = ASLNetOriginal(29)
    set_seed(0)
    b = ASLNetReLU(29)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    a.eval(), b.eval()
    x = torch.randn(4, 3, 100, 100)
    assert torch.allclose(a.features(x), b.features(x))
    assert not torch.allclose(a(x), b(x))


def test_gap_variant_shares_the_conv_stack_and_drops_the_flatten():
    """GAP changes only how the spatial map reaches the classifier."""
    set_seed(0)
    a = ASLNetOriginal(29)
    set_seed(0)
    g = ASLNetGAP(29)
    a.eval(), g.eval()
    x = torch.randn(4, 3, 100, 100)
    assert torch.allclose(a.features(x), g.features(x))     # identical conv stack
    assert a.f1.in_features == 2700 and g.f1.in_features == 27
    assert sum(p.numel() for p in g.parameters()) < sum(p.numel() for p in a.parameters()) / 20


def test_global_average_pooling_discards_position():
    """The mechanism claim: averaging over space is invariant to where a feature sits."""
    fmap = torch.randn(2, 27, 10, 10)
    pooled = fmap.mean(dim=(2, 3))
    for shift in (1, 3, 5):
        moved = torch.roll(fmap, shifts=(shift, shift), dims=(2, 3))
        assert torch.allclose(pooled, moved.mean(dim=(2, 3)), atol=1e-6)


def test_gradcam_shape_and_range():
    m = build_model("aslnet_original", n_classes=5, image_size=64)
    x = torch.randn(1, 3, 64, 64)
    with GradCAM(m, m.gradcam_layer) as cam:
        heat, idx = cam(x, class_idx=2)
    assert heat.shape == (64, 64), "heatmap must be resized to input resolution"
    assert 0.0 <= heat.min() and heat.max() <= 1.0 + 1e-6
    assert idx == 2


def test_gradcam_removes_hooks():
    m = build_model("aslnet_original", n_classes=5, image_size=64)
    before = len(m.gradcam_layer._forward_hooks)
    with GradCAM(m, m.gradcam_layer) as cam:
        cam(torch.randn(1, 3, 64, 64))
    assert len(m.gradcam_layer._forward_hooks) == before


def test_activation_maps_channel_count():
    m = build_model("aslnet_original", n_classes=5, image_size=100)
    acts = activation_maps(m, torch.randn(1, 3, 100, 100), m.gradcam_layer)
    assert acts.shape[0] == 27, "conv2 has 27 channels — the '27 activation maps' claim"


def test_robustness_suite_is_deterministic():
    suite = default_suite()
    assert len(suite) == 12
    assert {p.family for p in suite} == {"rotation", "brightness", "blur", "translation"}
    x = torch.rand(3, 32, 32)
    for p in suite:
        assert torch.allclose(p.fn(x), p.fn(x)), f"{p.name} is not deterministic"
