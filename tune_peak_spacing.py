"""Controlled peak-spacing experiment on a DEVELOPMENT plot only.

Defaults reproduce the plot-10 experiment. Ground is computed once. The reserved
plot must not be supplied here if its results are being used for validation.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from forest import load_cloud, normalise_heights, segment, evaluate_instances


def tune(input_path, config_path, output_dir, selected_path, kind='forinstance'):
    cfg = json.loads(Path(config_path).read_text())
    print('Loading development plot and calculating heights...', flush=True)
    xyz, cls, ref, _ = load_cloud(input_path, cfg['max_points'])
    if not np.any(ref > 0):
        raise ValueError('No positive reference tree IDs; labelled development data are required.')
    h, _ = normalise_heights(xyz, cls == cfg['ground_class'], cfg['terrain_cell_m'])
    valid = ~np.isin(cls, [0, 3]) if kind == 'forinstance' else np.ones(len(cls), bool)
    results, best_cfg, best_f1 = [], None, -1.0
    print('Spacing | Crowns | TP | FP | FN | Precision | Recall | F1', flush=True)
    for spacing in [3.0, 2.5, 2.0, 1.5, 1.0]:
        trial = dict(cfg, peak_separation_m=spacing)
        pred, trees, *_ = segment(xyz, h, trial)
        m = evaluate_instances(pred, ref, h, valid)
        results.append({'spacing_m': spacing, 'crowns': len(trees), **m})
        print(f"{spacing:7.1f} | {len(trees):6d} | {m['true_positives']:2d} | "
              f"{m['false_positives']:2d} | {m['false_negatives']:2d} | "
              f"{m['precision']:9.3f} | {m['recall']:6.3f} | {m['f1']:.3f}", flush=True)
        # Keep the first configuration when scores tie (larger spacing first).
        if m['f1'] > best_f1:
            best_f1, best_cfg = m['f1'], trial
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'peak_spacing_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    Path(selected_path).write_text(json.dumps(best_cfg, indent=2))
    (out / 'experiment.json').write_text(json.dumps({
        'input': str(input_path), 'dataset_kind': kind, 'base_config': cfg,
        'spacing_candidates_m': [3., 2.5, 2., 1.5, 1.],
        'selection_rule': 'maximum development point-IoU F1; retain first on ties',
        'selected_config_path': str(selected_path),
        'selected_spacing_m': best_cfg['peak_separation_m'],
        'development_f1': best_f1
    }, indent=2))
    print(f'Selected development configuration: {selected_path}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/forinstance/NIBIO/plot_10_annotated.las')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output', default='outputs/plot10_tuning')
    parser.add_argument('--selected-config', default='config_plot10_selected.json')
    parser.add_argument('--dataset-kind', choices=['forinstance', 'synthetic'], default='forinstance')
    parser.add_argument('--confirm-metre-units', action='store_true')
    args = parser.parse_args()
    if not args.confirm_metre_units:
        parser.error('Verify input XY and Z are in metres, then pass --confirm-metre-units.')
    try:
        tune(args.input, args.config, args.output, args.selected_config, args.dataset_kind)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'Tuning failed: {exc}')
