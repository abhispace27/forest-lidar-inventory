import numpy as np
import pytest
import json
from pathlib import Path
from forest import normalise_heights, segment, evaluate_instances, generate_demo, load_cloud


def test_sloping_ground_and_vertical_translation():
    x, y = np.meshgrid(np.linspace(0, 10, 21), np.linspace(0, 10, 21))
    ground = np.column_stack((x.ravel(), y.ravel(), (100 + .2*x + .1*y).ravel()))
    tree = np.array([[5.1, 4.3, 100 + .2*5.1 + .1*4.3 + 12]])
    xyz = np.vstack((ground, tree))
    mask = np.arange(len(xyz)) < len(ground)
    h, _ = normalise_heights(xyz, mask, .25)
    shifted = xyz.copy()
    shifted[:, 2] += 1000
    h2, _ = normalise_heights(shifted, mask, .25)
    assert h[-1] == pytest.approx(12, abs=1e-8)
    np.testing.assert_allclose(h, h2, atol=1e-9)


def test_refuse_missing_ground():
    with pytest.raises(ValueError, match='Too few ground'):
        normalise_heights(np.zeros((10, 3)), np.zeros(10, bool))


def test_strict_iou_and_one_to_one_matching():
    ref = np.ones(10, int)
    pred = np.array([1]*5 + [2]*5)
    report = evaluate_instances(pred, ref, np.ones(10), np.ones(10, bool))
    assert report['true_positives'] == 0  # Each split has IoU exactly 0.5.
    assert report['false_negatives'] == 1
    report = evaluate_instances(np.ones(10, int), ref, np.ones(10), np.ones(10, bool))
    assert report['true_positives'] == 1
    assert report['f1'] == 1


def test_unlabelled_outpoints_not_false_positives():
    pred = np.array([1, 1, 2, 2])
    ref = np.array([1, 1, 0, 0])
    valid = np.array([True, True, False, False])
    report = evaluate_instances(pred, ref, np.ones(4), valid)
    assert report['false_positives'] == 0
    assert report['f1'] == 1


def test_nine_synthetic_trees_end_to_end(tmp_path):
    path = tmp_path / 'synthetic.laz'
    generate_demo(path)
    xyz, cls, ref, _ = load_cloud(path)
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())
    h, _ = normalise_heights(xyz, cls == 2, cfg['terrain_cell_m'])
    pred, rows, *_ = segment(xyz, h, cfg)
    assert len(rows) == 9
    report = evaluate_instances(pred, ref, h, np.ones(len(xyz), bool))
    assert report['true_positives'] == 9
