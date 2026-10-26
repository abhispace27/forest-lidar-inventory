# Forest inventory from LiDAR: measured prototype results

An interpretable Python canopy-segmentation pipeline evaluated on two FOR-instance NIBIO plots.

## Experimental design

- Development: plot 10. Tested peak-spacing settings: 3.0 m, 2.5 m, 2.0 m, 1.5 m, 1.0 m.
- Selected spacing: 1 m, using development F1.
- Reserved evaluation: plot 6, with the selected configuration fixed before this evaluation.
- Both plots belong to the official development partition. Plot 6 is a reserved validation plot within that partition, not an official benchmark test result.
- Same collection, two plots only; no evidence of performance across sites or forest types.

## Results

| Run | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Plot 10 / development baseline | 3 | 6 | 22 | 33.3% | 12.0% | 0.176 |
| Plot 10 / development selected | 7 | 8 | 18 | 46.7% | 28.0% | 0.350 |
| Plot 6 / reserved baseline | 2 | 11 | 42 | 15.4% | 4.5% | 0.070 |
| Plot 6 / reserved selected | 9 | 12 | 35 | 42.9% | 20.5% | 0.277 |

Reserved-plot F1 changed from 0.070 to 0.277 (+0.207 absolute).
The selected run processed 4,551,358 points in 6.02 s, including HTML-view export; peak process RSS was 455 MB.

## Method

LAS/LAZ loading → supplied-ground terrain interpolation → height-above-ground raster → smoothed local maxima → marker-controlled watershed → per-segment inventory.

Evaluation uses one-to-one point-IoU matching with IoU strictly > 0.5. FOR-instance classes 0 and 3 are excluded from scoring. Outer points remain in inference, while predicted segments without points in the scored domain are ignored. All positive reference tree IDs in the scored domain count toward recall.

## Interpretation and limits

Reducing peak spacing improved reserved-plot matching from 2 to 9 trees. Recall remains 20.5%; many trees are still missed or do not meet the overlap threshold.
The result supports a useful improvement to this baseline on one reserved plot, not production readiness or general accuracy.
The detector uses supplied ground-class annotations, so this is not a raw-scan ground-classification pipeline.
Canopy projection discards vertical structure; overlapping crowns and suppressed trees require further diagnosis. Maximum-height rasterisation is sensitive to outliers.
Reported height RMSE measures agreement of cloud-derived p99 heights only on matched segments, using a shared terrain estimate. It is not independent field-height accuracy.
The selected plot is held in RAM. File reading is chunked; the analysis is not a streaming or distributed implementation.
Runtime is a single observed run on the recorded machine. Peak process RSS includes input, processing, imports and viewer export.

## Next technical priorities

1. Inspect reference-versus-prediction failures and stratify by canopy visibility and tree size.
2. Validate on additional reserved plots and collections with a predeclared evaluation protocol.
3. Add noise handling and bounded-memory spatial tiling with overlap reconciliation.

## Reproduce

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python forest.py run data/forinstance/NIBIO/plot_6_annotated.las --config config_plot10_selected.json --dataset-kind forinstance --confirm-metre-units --output outputs/nibio_plot6_selected
python make_report.py
```

See README.md for dataset download, baseline commands and annotation details. Keep both configurations and measured summaries with the project.

## Data attribution

Puliti et al., FOR-instance: a UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees.
Dataset: https://zenodo.org/records/8287792
Paper: https://arxiv.org/abs/2309.01279

This report contains derived aggregate metrics. Raw LiDAR is not included.
