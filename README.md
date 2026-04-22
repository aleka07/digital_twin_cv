# Observation Layer For Digital Twins In Manufacturing

This folder is the clean publishable core of the project: a vision-based observation layer for digital twins in manufacturing.

The main idea is simple:

- ingest production video from a fixed camera
- detect and track relevant entities
- derive observation variables such as occupancy, entries, exits, dwell time, trajectories, and heatmaps
- expose those variables as a lightweight observation interface for a manufacturing digital twin

This workspace keeps the framework-facing scripts, benchmark artifacts, notes, prompts, and evaluation outputs separated from older legacy experiments.

## What This Repository Contains

- `scripts/`
  Reproducible scripts for frame extraction, annotation review, detector training, tracker evaluation, zone analytics, and figure asset preparation.
- `results/`
  CSV summaries, benchmark notes, selected training artifacts, zone analytics, and demo outputs.
- `tracking_benchmark/`
  Short benchmark clips and tracking ground truth metadata.
- `zones/`
  Zone definitions and related review assets.
- `notes/`
  Experiment log, workflow notes, and design notes.
- `configs/`
  Evaluation YAML files for detector benchmarking.
- `observation_layer_prompt_files/`
  Expanded JSON prompts for framework diagrams and graphical-abstract style assets.

## Core Runtime Stack

- Detector: `YOLO11x`
- Tracker: `ByteTrack`
- Target classes: `person`, `vat`
- Observation variables:
  - occupancy
  - entries
  - exits
  - dwell time
  - trajectories
  - heatmaps

## Main Validation Assets

- Detector benchmark:
  - [results/detector_benchmark.csv](/c:/Users/SuperPC/Documents/DIPLOM/article_pack/results/detector_benchmark.csv)
- Tracker benchmark:
  - [results/tracker_benchmark.csv](/c:/Users/SuperPC/Documents/DIPLOM/article_pack/results/tracker_benchmark.csv)
- Zone analytics summary:
  - [results/zone_analytics_summary.md](/c:/Users/SuperPC/Documents/DIPLOM/article_pack/results/zone_analytics_summary.md)
- Long demonstration run:
  - [results/zone_demo/cam10_1700_fullhour_bytetrack/zone_demo_summary.md](/c:/Users/SuperPC/Documents/DIPLOM/article_pack/results/zone_demo/cam10_1700_fullhour_bytetrack/zone_demo_summary.md)
- Experiment log:
  - [notes/experiment_log.md](/c:/Users/SuperPC/Documents/DIPLOM/article_pack/notes/experiment_log.md)

## Installation

Create a Python environment and install:

```powershell
pip install -r article_pack\requirements.txt
```

## Typical Workflows

Detector training:

```powershell
python article_pack\scripts\train_seed_detector.py --model yolo11x.pt
```

Tracking benchmark:

```powershell
python article_pack\scripts\eval_tracking_benchmark.py
```

Zone analytics on annotated clips:

```powershell
python article_pack\scripts\run_zone_analytics.py
```

Long observation demo run:

```powershell
python article_pack\scripts\run_zone_demo_long.py `
  --video article_pack\data\cam10\source_video\1_10_R_23032026170000.mp4 `
  --zones article_pack\zones\cam10_zones_final.json `
  --model runs\detect\article_pack\experiments\detector\cam10_round12_yolo11x\weights\best.pt `
  --tracker bytetrack `
  --output-dir article_pack\results\zone_demo\cam10_1700_fullhour_bytetrack
```

## Open-Source Notes

This folder is already the best candidate for a future standalone GitHub repository, but a few release decisions still remain:

- final license choice
- whether to publish model weights directly or keep them as optional downloads
- how much manuscript-specific figure material should stay in the public repo
- whether to keep only summarized results or also selected demo assets

See:

- [OPEN_SOURCE_RELEASE_CHECKLIST.md](/c:/Users/SuperPC/Documents/DIPLOM/article_pack/OPEN_SOURCE_RELEASE_CHECKLIST.md)

## Scope Boundary

This workspace intentionally does **not** try to publish every historical file from the larger diploma folder.

The goal here is a clean framework-facing repository for:

- reusable observation-layer code
- reproducible benchmark summaries
- article and demonstration assets

