"""Small, interpretable canopy segmentation baseline. All distances are metres."""
from __future__ import annotations
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import laspy
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

ROOT = Path(__file__).resolve().parent


def load_cloud(path, max_points=15_000_000, crop=None):
    """Chunked input; selected plot is then held in memory. Does not stream analysis."""
    parts, classes, refs = [], [], []
    count = 0
    with laspy.open(path) as reader:
        dims = list(reader.header.point_format.dimension_names)
        ref_dim = next((d for d in dims if d.casefold() == 'treeid'), None)
        crs = reader.header.parse_crs()
        metadata = {'file': str(path), 'header_points': reader.header.point_count,
                    'dimensions': dims, 'crs': str(crs) if crs else None,
                    'bounds_min': reader.header.mins.tolist(),
                    'bounds_max': reader.header.maxs.tolist()}
        for chunk in reader.chunk_iterator(1_000_000):
            xyz = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(float)
            keep = np.isfinite(xyz).all(axis=1)
            if crop is not None:
                x0, y0, x1, y1 = crop
                keep &= ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1)
                         & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1))
            count += int(keep.sum())
            if count > max_points:
                raise ValueError('Selected plot exceeds max_points. Use --crop or deliberately raise config limit.')
            if keep.any():
                parts.append(xyz[keep])
                classes.append(np.asarray(chunk.classification)[keep])
                refs.append(np.asarray(chunk[ref_dim], dtype=np.int64)[keep]
                            if ref_dim else np.zeros(keep.sum(), dtype=np.int64))
    if not parts:
        raise ValueError('No finite points in selected area.')
    xyz, cls, ref = np.concatenate(parts), np.concatenate(classes), np.concatenate(refs)
    metadata['selected_points'] = len(xyz)
    metadata['class_counts'] = {str(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))}
    metadata['reference_tree_count'] = int(len(np.unique(ref[ref > 0])))
    return xyz, cls, ref, metadata


def normalise_heights(xyz, ground, cell_m=1.0):
    """Aggregate supplied ground points, then interpolate terrain in XY.

    Nearest-neighbour extrapolation outside the ground support is reported.
    Instance reference labels are never used here.
    """
    if ground.sum() < 6:
        raise ValueError('Too few ground points. Inspect classification codes; no silent flat-ground fallback.')
    g = xyz[ground]
    origin = g[:, :2].min(axis=0)
    ij = np.floor((g[:, :2] - origin) / cell_m).astype(np.int64)
    _, inverse = np.unique(ij, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    nodes = np.column_stack([np.bincount(inverse, weights=g[:, d]) / counts for d in range(3)])
    if len(nodes) > 100_000:
        raise ValueError('Too many terrain nodes. Crop the plot or increase terrain_cell_m.')
    if len(nodes) < 3 or np.linalg.matrix_rank(nodes[:, :2] - nodes[0, :2]) < 2:
        raise ValueError('Ground support must span a 2D area.')
    linear = LinearNDInterpolator(nodes[:, :2], nodes[:, 2])
    nearest = NearestNDInterpolator(nodes[:, :2], nodes[:, 2])
    terrain = np.empty(len(xyz), dtype=float)
    outside_count = 0
    for start in range(0, len(xyz), 250_000):
        xy = xyz[start:start + 250_000, :2]
        z = linear(xy)
        outside = ~np.isfinite(z)
        outside_count += int(outside.sum())
        z[outside] = nearest(xy[outside])
        terrain[start:start + len(xy)] = z
    return xyz[:, 2] - terrain, outside_count / len(xyz)


def segment(xyz, heights, cfg):
    origin = xyz[:, :2].min(axis=0)
    ij = np.floor((xyz[:, :2] - origin) / cfg['cell_m']).astype(np.int64)
    shape = (int(ij[:, 1].max()) + 1, int(ij[:, 0].max()) + 1)
    if shape[0] * shape[1] > cfg['max_grid_cells']:
        raise ValueError('Grid too large. Crop plot or increase cell_m; check coordinate units.')
    chm = np.full(shape, -np.inf)
    np.maximum.at(chm, (ij[:, 1], ij[:, 0]), heights)
    observed = np.isfinite(chm)
    # Empty pixels are excluded from the canopy mask. Normalised convolution
    # avoids treating missing observations as measured zero height.
    base = np.where(observed, np.maximum(chm, 0), 0)
    sigma = cfg['smooth_m'] / cfg['cell_m']
    if sigma > 0:
        num = ndi.gaussian_filter(base, sigma)
        den = ndi.gaussian_filter(observed.astype(float), sigma)
        smooth = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-8)
    else:
        smooth = base
    mask = observed & (base >= cfg['min_height_m'])
    peaks = peak_local_max(smooth, labels=mask.astype(np.uint8),
                           min_distance=max(1, int(np.ceil(cfg['peak_separation_m'] / cfg['cell_m']))),
                           threshold_abs=cfg['min_height_m'], exclude_border=False)
    markers = np.zeros(shape, dtype=np.int32)
    if len(peaks):
        markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1)
    labels = watershed(-smooth, markers, mask=mask) if len(peaks) else markers
    point_labels = labels[ij[:, 1], ij[:, 0]].copy()
    point_labels[heights < cfg['min_height_m']] = 0
    rows = []
    for tree_id, (r, c) in enumerate(peaks, 1):
        belongs = point_labels == tree_id
        if not belongs.any():
            continue
        rows.append({'tree_id': tree_id, 'treetop_x_m': float(origin[0] + (c + .5) * cfg['cell_m']),
                     'treetop_y_m': float(origin[1] + (r + .5) * cfg['cell_m']),
                     'height_p99_m': float(np.percentile(heights[belongs], 99)),
                     'crown_footprint_m2': float(np.sum(labels == tree_id) * cfg['cell_m'] ** 2),
                     'points_above_min_height': int(belongs.sum())})
    return point_labels, rows, base, labels, origin


def evaluate_instances(pred, ref, heights, valid, threshold=.5):
    """Point-IoU matching, strictly > threshold; class 3/0 excluded by caller.

    No field-height validation: height errors compare derived p99 heights of
    matched labelled and predicted point sets using the same terrain estimate.
    """
    pred, ref, h = pred[valid], ref[valid], heights[valid]
    pids, pc = np.unique(pred[pred > 0], return_counts=True)
    rids, rc = np.unique(ref[ref > 0], return_counts=True)
    if len(pids) * len(rids) > 4_000_000:
        raise ValueError('Evaluation matrix too large for this small-plot baseline.')
    intersections = np.zeros((len(pids), len(rids)), dtype=np.int64)
    both = (pred > 0) & (ref > 0)
    if both.any():
        np.add.at(intersections, (np.searchsorted(pids, pred[both]), np.searchsorted(rids, ref[both])), 1)
    union = pc[:, None] + rc[None, :] - intersections
    iou = np.divide(intersections, union, out=np.zeros_like(intersections, dtype=float), where=union > 0)
    if len(pids) and len(rids):
        reward = (iou > threshold) * (1 + iou / (min(len(pids), len(rids)) + 1))
        a, b = linear_sum_assignment(-reward)
        keep = iou[a, b] > threshold
        a, b = a[keep], b[keep]
    else:
        a = b = np.array([], dtype=int)
    tp, npred, nref = len(a), len(pids), len(rids)
    precision = tp / npred if npred else 0.0
    recall = tp / nref if nref else 0.0
    errors = [float(np.percentile(h[pred == pids[i]], 99) - np.percentile(h[ref == rids[j]], 99))
              for i, j in zip(a, b)]
    return {'protocol': 'point-IoU one-to-one matching; IoU strictly greater than 0.5',
            'reference_trees': nref, 'predicted_segments_in_scored_domain': npred,
            'true_positives': tp, 'false_positives': npred - tp, 'false_negatives': nref - tp,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'mean_matched_iou': float(iou[a, b].mean()) if tp else None,
            'matched_cloud_height_p99_rmse_m': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            'note': 'Height metric is cloud-derived agreement, not independent field-height accuracy.'}


def make_view(xyz, h, pred, chm, labels, origin, cfg, out, title):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}, {'type': 'xy'}, {'type': 'xy'}]],
                        subplot_titles=('3D points: detected crowns', 'Canopy height (m)', 'Crown footprints'))
    rng = np.random.default_rng(cfg['seed'])
    take = rng.choice(len(xyz), min(len(xyz), cfg['viewer_points']), replace=False)
    palette = ['#47b39c', '#e6b655', '#cd6973', '#7c98dc', '#b185cb', '#73c6d0', '#a5be6a']
    colours = ['#a9adb5' if i == 0 else palette[(i - 1) % len(palette)] for i in pred[take]]
    fig.add_trace(go.Scatter3d(x=xyz[take, 0] - origin[0], y=xyz[take, 1] - origin[1], z=h[take],
                              mode='markers', marker=dict(size=1.5, color=colours),
                              text=['Segment ' + str(i) for i in pred[take]],
                              hovertemplate='%{text}<br>x=%{x:.1f} m<br>y=%{y:.1f} m<br>HAG=%{z:.1f} m<extra></extra>',
                              showlegend=False), row=1, col=1)
    # Downsample display rasters only; processing and CSV use full resolution.
    stride = max(1, int(np.ceil(max(chm.shape) / 700)))
    xs = (np.arange(chm.shape[1])[::stride] + .5) * cfg['cell_m']
    ys = (np.arange(chm.shape[0])[::stride] + .5) * cfg['cell_m']
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=chm[::stride, ::stride], colorscale='Viridis', showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=labels[::stride, ::stride], colorscale='Turbo', showscale=False), row=1, col=3)
    fig.update_layout(title=title, height=650, template='plotly_white',
                      scene=dict(xaxis_title='Local x (m)', yaxis_title='Local y (m)', zaxis_title='Height above ground (m)', aspectmode='data'),
                      margin=dict(l=25, r=25, t=100, b=40))
    for col in (2, 3):
        fig.update_xaxes(title_text='Local x (m)', row=1, col=col)
        fig.update_yaxes(title_text='Local y (m)', scaleanchor='x' if col == 2 else 'x2', row=1, col=col)
    fig.write_html(out / 'viewer.html', include_plotlyjs=True, full_html=True)


def generate_demo(path, seed=40):
    """Nine separated artificial conifers on sloping terrain. Not real LiDAR."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.linspace(0, 40, 161), np.linspace(0, 40, 161))
    ground = np.column_stack((gx.ravel(), gy.ravel(), (100 + .10 * gx + .05 * gy).ravel()))
    parts, classes, refs = [ground], [np.full(len(ground), 2)], [np.zeros(len(ground), dtype=int)]
    for tree_id, (cx, cy) in enumerate([(x, y) for x in (8, 20, 32) for y in (8, 20, 32)], 1):
        n = 7000
        radius = 4.0 * np.sqrt(rng.random(n))
        angle = rng.uniform(0, 2 * np.pi, n)
        x, y = cx + radius * np.cos(angle), cy + radius * np.sin(angle)
        height = (16 + tree_id * .5) * (1 - radius / 4) + rng.normal(0, .08, n)
        height = np.maximum(height, .2)
        z = 100 + .10 * x + .05 * y + height
        parts.append(np.column_stack((x, y, z)))
        classes.append(np.full(n, 5))
        refs.append(np.full(n, tree_id))
    xyz = np.concatenate(parts)
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = np.array([.001, .001, .001])
    header.offsets = xyz.min(axis=0)
    header.add_extra_dim(laspy.ExtraBytesParams(name='treeID', type=np.int32))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz.T
    cloud.classification = np.concatenate(classes).astype(np.uint8)
    cloud.treeID = np.concatenate(refs).astype(np.int32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.write(path)
    print(f'Created SYNTHETIC smoke-test forest: {path} (9 trees)')


def run(args):
    start = time.perf_counter()
    cfg = json.loads(Path(args.config).read_text())
    if any(cfg[k] <= 0 for k in ('cell_m', 'terrain_cell_m', 'peak_separation_m', 'min_height_m')) or cfg['smooth_m'] < 0:
        raise ValueError('Spatial parameters must be positive; smooth_m may be zero.')
    if not args.confirm_metre_units:
        raise ValueError('Verify XY and Z are in metres, then pass --confirm-metre-units.')
    xyz, cls, ref, metadata = load_cloud(args.input, cfg['max_points'], args.crop)
    h, fraction = normalise_heights(xyz, cls == cfg['ground_class'], cfg['terrain_cell_m'])
    pred, rows, chm, labels, origin = segment(xyz, h, cfg)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fields = ['tree_id', 'treetop_x_m', 'treetop_y_m', 'height_p99_m', 'crown_footprint_m2', 'points_above_min_height']
    with (out / 'trees.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {'dataset_kind': args.dataset_kind, 'input': metadata, 'config': cfg,
               'crop': args.crop, 'detected_crowns': len(rows),
               'terrain_extrapolated_point_fraction': fraction,
               'ground_source': 'supplied classification labels (not predicted by this pipeline)',
               'python': platform.python_version(), 'platform': platform.platform(),
               'limitations': ['2.5D canopy baseline; can miss understory or merge/split crowns.',
                               'Height p99 and crown footprints are estimates from assigned points.',
                               'Full selected plot is held in memory; chunked input is not streaming analysis.']}
    if args.dataset_kind in ('forinstance', 'synthetic') and np.any(ref > 0):
        valid = ~np.isin(cls, [0, 3]) if args.dataset_kind == 'forinstance' else np.ones(len(cls), dtype=bool)
        summary['evaluation'] = evaluate_instances(pred, ref, h, valid)
    else:
        summary['evaluation'] = {'status': 'not evaluated; supply FOR-instance labels and correct dataset-kind'}
    title = ('SYNTHETIC SETUP CHECK — ' if args.dataset_kind == 'synthetic' else 'Forest canopy baseline — ') + Path(args.input).name
    make_view(xyz, h, pred, chm, labels, origin, cfg, out, title)
    summary['wall_time_s_including_view_export'] = time.perf_counter() - start
    try:
        import resource
        summary['peak_process_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != 'darwin' else 1024 ** 2)
    except ImportError:
        summary['peak_process_rss_mb'] = None
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'config_used.json').write_text(json.dumps(cfg, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Open: {(out / "viewer.html").resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Generate a SYNTHETIC LAS setup check')
    demo.add_argument('--output', default='data/synthetic.las')
    inspect = sub.add_parser('inspect', help='Inspect one LAS/LAZ plot before processing')
    inspect.add_argument('input')
    inspect.add_argument('--max-points', type=int, default=15_000_000)
    process = sub.add_parser('run', help='Create inventory, self-contained viewer and evaluation')
    process.add_argument('input')
    process.add_argument('--config', default=str(ROOT / 'config.json'))
    process.add_argument('--output', default='outputs/run')
    process.add_argument('--dataset-kind', choices=['forinstance', 'synthetic', 'other'], default='other')
    process.add_argument('--confirm-metre-units', action='store_true')
    process.add_argument('--crop', nargs=4, type=float, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'))
    args = parser.parse_args()
    try:
        if args.command == 'demo':
            generate_demo(args.output)
        elif args.command == 'inspect':
            print(json.dumps(load_cloud(args.input, args.max_points)[3], indent=2))
        else:
            if args.crop and (args.crop[0] >= args.crop[2] or args.crop[1] >= args.crop[3]):
                raise ValueError('Crop min must be below max.')
            run(args)
    except (ValueError, OSError, laspy.errors.LaspyException) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
