"""Build an offline project report from existing run outputs; no new fitting.

Place beside forest.py and run: python make_report.py
Outputs: report/index.html, report/RESULTS.md, report/metrics.csv,
report/tuning.csv and report/environment.json. Raw LiDAR is never copied.
"""
from __future__ import annotations
import csv
import html
import importlib.metadata
import io
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_run(root, folder):
    path = root / 'outputs' / folder / 'summary.json'
    if not path.exists():
        raise ValueError(f'Missing {path}. Run that experiment first.')
    data = json.loads(path.read_text())
    for field in ('input', 'config', 'evaluation', 'wall_time_s_including_view_export', 'peak_process_rss_mb'):
        if field not in data:
            raise ValueError(f'{path} lacks {field}.')
    evaluation = data['evaluation']
    for key in ('reference_trees', 'true_positives', 'false_positives', 'false_negatives', 'precision', 'recall', 'f1'):
        if key not in evaluation:
            raise ValueError(f'{path} lacks evaluated metric {key}.')
    return data


def pct(value):
    return f'{100 * float(value):.1f}%'


def score_row(label, m, points='', seconds='', memory=''):
    return {'run': label, 'reference_trees': int(m['reference_trees']),
            'true_positives': int(m['true_positives']),
            'false_positives': int(m['false_positives']),
            'false_negatives': int(m['false_negatives']),
            'precision': float(m['precision']), 'recall': float(m['recall']), 'f1': float(m['f1']),
            'input_points': points, 'wall_time_s': seconds, 'peak_process_rss_mb': memory}


def build_report(root=ROOT):
    root = Path(root)
    b10 = read_run(root, 'nibio_plot10_baseline')
    b6 = read_run(root, 'nibio_plot6_baseline')
    s6 = read_run(root, 'nibio_plot6_selected')
    tuning_path = root / 'outputs' / 'plot10_tuning' / 'peak_spacing_results.csv'
    if not tuning_path.exists():
        raise ValueError(f'Missing {tuning_path}. Run the peak-spacing experiment first.')
    with tuning_path.open(newline='') as f:
        tuning = list(csv.DictReader(f))
    if not tuning:
        raise ValueError('Tuning results are empty.')
    if b6['input']['file'] != s6['input']['file']:
        raise ValueError('Plot 6 baseline and selected run must use the same input.')
    if b6.get('crop') != s6.get('crop'):
        raise ValueError('Plot 6 evaluated crops differ.')
    changed = {k for k in set(b6['config']) | set(s6['config'])
               if b6['config'].get(k) != s6['config'].get(k)}
    if changed - {'peak_separation_m'}:
        raise ValueError(f'Additional parameters changed: {sorted(changed)}. Describe these before reporting a spacing-only experiment.')
    spacing = float(s6['config']['peak_separation_m'])
    matches = [r for r in tuning if abs(float(r['spacing_m']) - spacing) < 1e-9]
    if not matches:
        raise ValueError('Selected plot 6 spacing has no corresponding development result.')
    selected_dev = matches[0]
    best_f1 = max(float(r['f1']) for r in tuning)
    if abs(float(selected_dev['f1']) - best_f1) > 1e-9:
        raise ValueError('Selected spacing is not the best recorded development F1. Document the alternative selection rule first.')
    for k, value in b10['config'].items():
        if k != 'peak_separation_m' and value != s6['config'].get(k):
            raise ValueError('Development and reserved-plot configurations differ beyond spacing.')
    rows = []
    for label, run in [('Plot 10 / development baseline', b10),
                       ('Plot 6 / reserved baseline', b6),
                       ('Plot 6 / reserved selected', s6)]:
        rows.append(score_row(label, run['evaluation'], run['input']['selected_points'],
                              run['wall_time_s_including_view_export'], run['peak_process_rss_mb']))
    rows.insert(1, score_row('Plot 10 / development selected', selected_dev))
    e0, e1 = b6['evaluation'], s6['evaluation']
    delta = float(e1['f1']) - float(e0['f1'])
    ratio = float(e1['f1']) / float(e0['f1']) if float(e0['f1']) > 0 else None
    baseline_spacing = b6['config']['peak_separation_m']
    parameters = ', '.join(f"{r['spacing_m']} m" for r in tuning)
    n = s6['input']['selected_points']
    runtime = s6['wall_time_s_including_view_export']
    rss = s6['peak_process_rss_mb']
    out = root / 'report'
    out.mkdir(exist_ok=True)
    with (out / 'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # Preserve the actual recorded development table, not reconstructed rounded scores.
    (out / 'tuning.csv').write_text(tuning_path.read_text())
    packages = {}
    for package in ('numpy', 'scipy', 'scikit-image', 'laspy', 'lazrs', 'plotly', 'pyproj', 'pytest'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    (out / 'environment.json').write_text(json.dumps({
        'report_python': platform.python_version(), 'report_platform': platform.platform(),
        'run_python': s6.get('python'), 'run_platform': s6.get('platform'),
        'installed_packages_at_report_creation': packages,
        'selected_config': s6['config'],
        'note': 'Report-generation environment may differ from the run environment.'
    }, indent=2))
    lines = ['# Forest inventory from LiDAR: measured prototype results', '',
             'An interpretable Python canopy-segmentation pipeline evaluated on two FOR-instance NIBIO plots.', '',
             '## Experimental design', '',
             f'- Development: plot 10. Tested peak-spacing settings: {parameters}.',
             f'- Selected spacing: {spacing:g} m, using development F1.',
             '- Reserved evaluation: plot 6, with the selected configuration fixed before this evaluation.',
             '- Both plots belong to the official development partition. Plot 6 is a reserved validation plot within that partition, not an official benchmark test result.',
             '- Same collection, two plots only; no evidence of performance across sites or forest types.', '',
             '## Results', '',
             '| Run | TP | FP | FN | Precision | Recall | F1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['run']} | {r['true_positives']} | {r['false_positives']} | {r['false_negatives']} | {pct(r['precision'])} | {pct(r['recall'])} | {r['f1']:.3f} |")
    lines += ['', f"Reserved-plot F1 changed from {e0['f1']:.3f} to {e1['f1']:.3f} (+{delta:.3f} absolute).",
              f"The selected run processed {n:,} points in {runtime:.2f} s, including HTML-view export; peak process RSS was {rss:.0f} MB." if rss is not None else f"The selected run processed {n:,} points in {runtime:.2f} s, including HTML-view export.",
              '', '## Method', '',
              'LAS/LAZ loading → supplied-ground terrain interpolation → height-above-ground raster → smoothed local maxima → marker-controlled watershed → per-segment inventory.', '',
              'Evaluation uses one-to-one point-IoU matching with IoU strictly > 0.5. FOR-instance classes 0 and 3 are excluded from scoring. Outer points remain in inference, while predicted segments without points in the scored domain are ignored. All positive reference tree IDs in the scored domain count toward recall.', '',
              '## Interpretation and limits', '',
              f"Reducing peak spacing improved reserved-plot matching from {e0['true_positives']} to {e1['true_positives']} trees. Recall remains {pct(e1['recall'])}; many trees are still missed or do not meet the overlap threshold.",
              'The result supports a useful improvement to this baseline on one reserved plot, not production readiness or general accuracy.',
              'The detector uses supplied ground-class annotations, so this is not a raw-scan ground-classification pipeline.',
              'Canopy projection discards vertical structure; overlapping crowns and suppressed trees require further diagnosis. Maximum-height rasterisation is sensitive to outliers.',
              'Reported height RMSE measures agreement of cloud-derived p99 heights only on matched segments, using a shared terrain estimate. It is not independent field-height accuracy.',
              'The selected plot is held in RAM. File reading is chunked; the analysis is not a streaming or distributed implementation.',
              'Runtime is a single observed run on the recorded machine. Peak process RSS includes input, processing, imports and viewer export.', '',
              '## Next technical priorities', '',
              '1. Inspect reference-versus-prediction failures and stratify by canopy visibility and tree size.',
              '2. Validate on additional reserved plots and collections with a predeclared evaluation protocol.',
              '3. Add noise handling and bounded-memory spatial tiling with overlap reconciliation.', '',
              '## Reproduce', '', 'From the repository root:', '', '```bash',
              'python -m pip install -r requirements.txt', 'python -m pytest -q',
              'python forest.py run data/forinstance/NIBIO/plot_6_annotated.las --config config_plot10_selected.json --dataset-kind forinstance --confirm-metre-units --output outputs/nibio_plot6_selected',
              'python make_report.py', '```', '',
              'See README.md for dataset download, baseline commands and annotation details. Keep both configurations and measured summaries with the project.', '',
              '## Data attribution', '',
              'Puliti et al., FOR-instance: a UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees.',
              'Dataset: https://zenodo.org/records/8287792', 'Paper: https://arxiv.org/abs/2309.01279', '',
              'This report contains derived aggregate metrics. Raw LiDAR is not included.', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    table_rows = ''.join(
        '<tr><td>' + html.escape(r['run']) + '</td>' +
        ''.join(f'<td>{r[k]}</td>' for k in ('true_positives', 'false_positives', 'false_negatives')) +
        f"<td>{pct(r['precision'])}</td><td>{pct(r['recall'])}</td><td><b>{r['f1']:.3f}</b></td></tr>"
        for r in rows)
    tuning_rows = ''.join(f"<tr><td>{float(r['spacing_m']):g} m</td><td>{int(r['crowns'])}</td><td>{int(r['true_positives'])}</td><td>{float(r['f1']):.3f}</td></tr>" for r in tuning)
    memory = f'{rss:.0f} MB' if rss is not None else 'Not measured'
    relative = f'{ratio:.2f}×' if ratio is not None else '—'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forest inventory | Prototype results</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f2;color:#172c27;font:16px/1.6 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:56px 26px}h1{font-size:clamp(32px,5vw,52px);line-height:1.12;letter-spacing:-1.5px;margin:14px 0 18px}h2{font-size:23px;margin:0 0 16px}p{max-width:830px}.eyebrow{letter-spacing:2px;font-size:12px;font-weight:700;text-transform:uppercase;color:#346c57}.lead{font-size:19px;color:#49625a}.pill{display:inline-block;padding:5px 12px;border-radius:30px;background:#e2ece3;font-size:13px;margin-right:8px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:30px 0}.card,section{background:white;border:1px solid #dce5dc;border-radius:14px;padding:24px}.number{font-size:34px;font-weight:700;letter-spacing:-1px}.muted,small{color:#607068}.small{font-size:14px}.flow{font-size:17px;font-weight:600;color:#346c57}section{margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:right;padding:12px 10px;border-bottom:1px solid #e4eae3;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{font-size:12px;text-transform:uppercase;letter-spacing:.4px;color:#607068}tbody tr:last-child{background:#f1f7ee}a{color:#21614b;text-underline-offset:4px}.button{display:inline-block;background:#215e49;color:white;text-decoration:none;padding:11px 18px;border-radius:7px;margin-right:8px}.notice{border-left:4px solid #b7c798;padding-left:18px}footer{font-size:13px;color:#607068;margin-top:30px}@media(max-width:650px){.cards{grid-template-columns:1fr}main{padding:30px 16px}section{padding:18px}}
</style><main><div class="eyebrow">Abhinav Kumar / Scientific software prototype</div>
<h1>From forest measurements<br>to a tree inventory.</h1><p class="lead">An interpretable LiDAR pipeline with measured errors, a controlled parameter experiment and a reserved-plot evaluation.</p>
<span class="pill">Python</span><span class="pill">FOR-instance / NIBIO</span><span class="pill">Two-plot study</span>
<div class="cards"><div class="card"><div class="muted">Reserved-plot F1</div><div class="number">@@F0@@ → @@F1@@</div><small>@@RATIO@@ relative; +@@DELTA@@ absolute</small></div><div class="card"><div class="muted">Selected run</div><div class="number">@@SECONDS@@ s</div><small>@@POINTS@@ points, including viewer export</small></div><div class="card"><div class="muted">Peak process memory</div><div class="number">@@MEMORY@@</div><small>Single run on the recorded local machine</small></div></div>
<section><h2>A deliberately simple baseline</h2><p class="flow">LAS points → terrain correction → canopy heights → treetop peaks → watershed crowns → inventory</p><p>Uses supplied ground labels. Reference tree IDs are kept out of detection and used only for evaluation. The 2.5D canopy projection keeps computation simple but loses vertical forest structure.</p></section>
<section><h2>Did the improvement transfer?</h2><p>Peak spacing was selected on plot 10, then fixed for plot 6. Both plots come from the official development partition; this is a reserved validation check within one collection.</p><div class="scroll"><table><thead><tr><th>Run</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>@@RESULTS@@</tbody></table></div><p class="small muted">Matching uses point intersection-over-union strictly greater than 0.5. Unlabelled outer points are excluded from scoring. These values are exploratory results, not an official benchmark submission.</p></section>
<section><h2>The controlled experiment</h2><p>Only peak separation changed from @@BASELINE_SPACING@@ m to @@SELECTED_SPACING@@ m; raster resolution, smoothing and height threshold stayed fixed.</p><div class="scroll"><table><thead><tr><th>Spacing</th><th>All detected crowns</th><th>Matched trees</th><th>Development F1</th></tr></thead><tbody>@@TUNING@@</tbody></table></div><p class="small muted">All-crown counts include detections outside the scored region and therefore differ from the counts used to compute precision.</p></section>
<section><h2>Inspect the actual outputs</h2><p>Rotate the measured point cloud, inspect crown footprints and compare the baseline with the selected run.</p><a class="button" href="../outputs/nibio_plot6_selected/viewer.html">Selected 3D viewer</a><a href="../outputs/nibio_plot6_baseline/viewer.html">Baseline viewer</a><p class="small muted">Viewer links require the existing outputs folder beside this report. This report itself opens offline.</p></section>
<section><h2>What remains unresolved</h2><div class="notice"><p>Selected reserved-plot recall is <b>@@RECALL@@</b>: <b>@@MISSED@@ of @@REFS@@ reference trees</b> remain unmatched. This is a useful baseline with a demonstrated improvement; substantial segmentation work remains.</p></div><ul><li>Canopy projection can merge neighbouring trees or miss suppressed crowns.</li><li>Ground classification is supplied; heights depend on the terrain estimate.</li><li>Height agreement is evaluated only for matched trees and is not independent field validation.</li><li>Two plots in one collection do not establish cross-site generalisation.</li><li>Analysis holds the selected plot in memory; scaling needs spatial tiling and reconciliation.</li></ul></section>
<footer>Metrics are generated from saved run outputs. <a href="RESULTS.md">Detailed results</a> · <a href="metrics.csv">Metrics CSV</a> · <a href="environment.json">Environment and configuration</a><br>Data: Puliti et al., <a href="https://zenodo.org/records/8287792">FOR-instance</a>. Raw LiDAR is not included in this report.</footer></main></html>'''
    replacements = {'F0': f"{e0['f1']:.3f}", 'F1': f"{e1['f1']:.3f}", 'RATIO': relative,
                    'DELTA': f'{delta:.3f}', 'SECONDS': f'{runtime:.2f}', 'POINTS': f'{n:,}',
                    'MEMORY': memory, 'RESULTS': table_rows, 'TUNING': tuning_rows,
                    'BASELINE_SPACING': f'{baseline_spacing:g}', 'SELECTED_SPACING': f'{spacing:g}',
                    'RECALL': pct(e1['recall']), 'MISSED': str(e1['false_negatives']), 'REFS': str(e1['reference_trees'])}
    for key, value in replacements.items():
        page = page.replace('@@' + key + '@@', value)
    (out / 'index.html').write_text(page)
    print(f'Report: {out / "index.html"}')
    print(f'Detailed results: {out / "RESULTS.md"}')
    print('Existing runs and configurations were not modified.')
    return out


if __name__ == '__main__':
    try:
        build_report()
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f'Report not generated: {exc}')
