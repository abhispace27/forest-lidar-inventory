# Forest LiDAR inventory

An interpretable Python prototype that turns a small forest LAS/LAZ scan into canopy segments, estimated heights, a CSV inventory and an interactive 3D view. A controlled parameter experiment improved point-IoU F1 on a reserved NIBIO plot from **0.070 to 0.277**, with a selected-run runtime of **6.02 s for 4.55 million points** on the development machine.

**Scope:** a two-plot study using FOR-instance's supplied ground labels. Segmentation remains modest: the selected configuration matches **9 of 44 reference trees** on the reserved plot. This is a reproducible scientific software prototype, not a production forestry service or an official benchmark submission.

[Measured results](report/RESULTS.md) · [Metrics CSV](report/metrics.csv) · [Report source](report/index.html)

Download the repository and open `report/index.html` to explore the report locally. GitHub displays HTML as source; interactive viewing requires a local browser or GitHub Pages. On Pages, the report is at `<repository-pages-url>/report/`.

## Results

| Plot and configuration | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Plot 10: development, original 3 m spacing | 3 | 6 | 22 | 33.3% | 12.0% | 0.176 |
| Plot 10: development, selected 1 m spacing | 7 | 8 | 18 | 46.7% | 28.0% | 0.350 |
| Plot 6: reserved validation, original settings | 2 | 11 | 42 | 15.4% | 4.5% | 0.070 |
| Plot 6: reserved validation, selected settings | 9 | 12 | 35 | 42.9% | 20.5% | 0.277 |

Spacing was selected on plot 10 from 3, 2.5, 2, 1.5 and 1 m, with all other parameters fixed. The chosen configuration was then evaluated on plot 6. Both plots belong to the dataset's official **development partition**; plot 6 is a reserved validation plot within it. The experiment demonstrates improvement on one reserved plot in the same collection, not cross-site generalisation.

The selected plot-6 run took 6.02 s including input, processing, evaluation and HTML export, with about 455 MB peak process RSS on the local Ubuntu laptop / Python 3.12.3. These are single-run observations, not an averaged performance benchmark. The captured platform and environment are in `report/environment.json`; detailed CPU/storage specifications were not recorded.

## Pipeline

1. Read LAS/LAZ data in chunks; retain the selected plot in memory.
2. Aggregate supplied ground-class points and interpolate terrain in XY.
3. Subtract terrain elevation to obtain height above ground.
4. Rasterise canopy heights, smooth the raster and detect local peaks.
5. Use marker-controlled watershed to estimate canopy crown footprints.
6. Assign above-threshold 3D points by their footprint and export per-segment measurements.
7. Evaluate against reference tree IDs, which are never inputs to detection.

The 2.5D representation keeps the baseline simple and fast. Its cost is losing vertical structure: overlapping crowns and suppressed trees remain difficult.

### Output definitions

- **Location:** centre of the detected peak's raster cell, not a verified stem position.
- **Height:** p99 of above-threshold points assigned to a segment.
- **Crown area:** projected raster footprint, not canopy surface area.
- **Matching:** one-to-one point-IoU matching, strictly greater than 0.5. FOR-instance class 0 and unlabelled outer class 3 are excluded from scoring. Outer points remain available during inference; segments without scored points are ignored.
- **Height RMSE:** agreement of cloud-derived p99 heights for matched point sets using the same terrain estimate. It is not independent field-height validation and does not describe unmatched trees.

## Install

Python 3.10–3.12 is recommended for this starter. On Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

`requirements.txt` gives supported dependency ranges. `requirements-lock.txt`, when included, records the exact packages from the development environment; it is a pip freeze snapshot, not a cross-platform lockfile. Use the dependency ranges for the CI Python version.

If Ubuntu reports missing `ensurepip`, install the matching venv package (usually `sudo apt install python3-venv`) and retry. Use the activated terminal initially to avoid differences between terminal and Spyder environments.

## Synthetic setup check

```bash
python forest.py demo
python forest.py run data/synthetic.las --dataset-kind synthetic --confirm-metre-units --output outputs/synthetic
xdg-open outputs/synthetic/viewer.html
```

The generated scene has nine separated artificial cone-shaped trees on sloping terrain. It checks installation and geometry; it is deliberately easy and is not a sensor simulation or evidence of real forest accuracy.

## Get real data

Download `FORinstance_dataset.zip` (about 1.6 GB) from [Zenodo](https://zenodo.org/records/8287792). From the repository root, if the archive is in the parent folder:

```bash
python -m zipfile -e ../FORinstance_dataset.zip data/forinstance
python forest.py inspect data/forinstance/NIBIO/plot_10_annotated.las
```

Read `data_split_metadata.csv` before choosing plots. Verify XY and Z units before using metric parameters. The two NIBIO plots used here report EPSG:25832 for horizontal coordinates and metre-scale elevation values. FOR-instance uses classification 2 for terrain and `treeID` for tree instances (paper Table 3); do not assume its other semantic codes apply to unrelated LAS datasets.

Raw scans are excluded from Git. Consult the dataset's attribution and redistribution terms for any use of its data and derivatives.

## Reproduce the experiment

Run the original development baseline:

```bash
python forest.py run data/forinstance/NIBIO/plot_10_annotated.las --config config.json --dataset-kind forinstance --confirm-metre-units --output outputs/nibio_plot10_baseline
```

Select spacing on development plot 10:

```bash
python tune_peak_spacing.py --confirm-metre-units
```

Compare the original and selected configurations on reserved plot 6:

```bash
python forest.py run data/forinstance/NIBIO/plot_6_annotated.las --config config.json --dataset-kind forinstance --confirm-metre-units --output outputs/nibio_plot6_baseline
python forest.py run data/forinstance/NIBIO/plot_6_annotated.las --config config_plot10_selected.json --dataset-kind forinstance --confirm-metre-units --output outputs/nibio_plot6_selected
```

Generate the report:

```bash
python make_report.py
xdg-open report/index.html
```

The tuning script loads and normalises the development data once. It records the five candidate results and selected configuration. The report generator reads saved outputs; it performs no fitting or parameter selection on the reserved plot.

## Files and engineering checks

| File | Purpose |
|---|---|
| `forest.py` | Input, terrain correction, segmentation, evaluation and visualisation functions; CLI entry point |
| `tune_peak_spacing.py` | Reproducible, controlled development experiment |
| `config.json` | Original parameters |
| `config_plot10_selected.json` | Parameters selected on plot 10 |
| `make_report.py` | Offline HTML report and machine-readable results |
| `tests/test_pipeline.py` | Five numerical and integration tests |
| `.github/workflows/checks.yml` | Test workflow for pushes and pull requests |
| `report/` | Measured metrics, report and environment record |

Tests check terrain correction on a known sloping plane, invariance to vertical translation, explicit missing-ground failure, strict IoU and one-to-one matching, exclusion of unlabelled outer points, and a nine-tree LAS/LAZ integration case. There are five test functions, some covering multiple checks. The GitHub workflow must run successfully on the published repository before CI status is claimed.

The loader limits selected data to 15 million points. Large plots require deliberate cropping or a reviewed limit increase. Cropping can truncate crowns and changes the evaluated domain. Terrain extrapolation is recorded; inspect it before interpreting heights. Chunked I/O does not make the analysis bounded-memory or distributed.

## Limitations and next priorities

- Recall remains low, and segmentation failures need reference-versus-prediction diagnosis.
- Ground classifications are supplied rather than predicted from raw scans.
- Maximum-height rasterisation is sensitive to high outliers.
- Two nearby plots from one collection do not establish performance across forests or acquisition conditions.
- Source point geometry and environment affect results; preserve coordinates, units and configurations.

Next priorities are to stratify errors by tree size and canopy visibility, test additional reserved collections with a predeclared protocol, and add spatial tiling with overlap reconciliation. Cloud deployment and learned 3D segmentation would follow measured need.

## Attribution

Puliti, S., Pearse, G., Surový, P., Wallace, L., Hollaus, M., Wielgosz, M., and Astrup, R. *FOR-instance: a UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees.*

[Dataset](https://zenodo.org/records/8287792) · [Paper](https://arxiv.org/abs/2309.01279)
