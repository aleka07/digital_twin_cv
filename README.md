# Observation Layer For Digital Twins In Manufacturing

A vision-based observation framework for manufacturing digital twins, built around production-video perception, tracking, spatial semantics, and derived observation variables.

## Overview

This repository implements an observation-level computer-vision stack for manufacturing digital twins.

The framework turns production video into structured observation variables that can support monitoring, analytics, and decision support:

- object detection
- multi-object tracking
- zone-aware spatial semantics
- occupancy estimation
- entries and exits
- dwell time
- trajectories
- spatial heatmaps

The current validated runtime stack in this repository is:

- Detector: `YOLO11x`
- Tracker: `ByteTrack`
- Target classes: `person`, `vat`

## Repository Structure

- `scripts/`
  Reproducible scripts for data preparation, annotation review, detector training, tracker benchmarking, zone analytics, and figure-asset preparation.
- `results/`
  Detector and tracker benchmarks, selected detector training artifacts, zone analytics summaries, and long-run demonstration outputs.
- `tracking_benchmark/`
  Benchmark metadata, draft tracker outputs, manifests, and tracking ground truth.
- `zones/`
  Zone definitions and supporting zone-review assets.
- `notes/`
  Experiment log, annotation workflow notes, and design notes.
- `configs/`
  Detector evaluation YAML files.
- `tracking_configs/`
  Tracker configuration variants used during experimentation.
- `observation_layer_prompt_files/`
  Expanded JSON prompt files for framework and graphical-abstract figures.

## Main Validation Assets

- Detector benchmark:
  - [results/detector_benchmark.csv](results/detector_benchmark.csv)
- Tracker benchmark:
  - [results/tracker_benchmark.csv](results/tracker_benchmark.csv)
- Tracking details:
  - [results/tracking_benchmark_details.md](results/tracking_benchmark_details.md)
- Zone analytics summary:
  - [results/zone_analytics_summary.md](results/zone_analytics_summary.md)
- Long demonstration run:
  - [results/zone_demo/cam10_1700_fullhour_bytetrack/zone_demo_summary.md](results/zone_demo/cam10_1700_fullhour_bytetrack/zone_demo_summary.md)
- Experiment log:
  - [notes/experiment_log.md](notes/experiment_log.md)

## Installation

Create a Python environment and install the dependencies:

```powershell
pip install -r requirements.txt
```

## Quick Start

Run detector training:

```powershell
python scripts/train_seed_detector.py --model yolo11x.pt
```

Run the tracking benchmark:

```powershell
python scripts/eval_tracking_benchmark.py
```

Run zone analytics on annotated clips:

```powershell
python scripts/run_zone_analytics.py
```

Run the long observation demo:

```powershell
python scripts/run_zone_demo_long.py `
  --video <path-to-production-video.mp4> `
  --zones zones/cam10_zones_final.json `
  --model <path-to-detector-weights.pt> `
  --tracker bytetrack `
  --output-dir results/zone_demo/example_long_run
```

## Reproducibility

The repository is designed to preserve the main framework logic and compact reproducibility assets.

Included:

- framework scripts
- benchmark CSV and markdown summaries
- compact selected detector-training artifacts
- tracking benchmark metadata and GT files
- zone definitions and zone analytics outputs
- article-oriented figure prompt assets

Excluded by default:

- raw private or heavy source videos
- heavyweight local frame dumps
- local archives
- optional large model weights
- temporary frame-selection assets

## Code Availability

This repository serves as the public codebase for the manuscript on an observation-level framework for digital twins in manufacturing. It contains the main implementation modules, detector and tracker evaluation scripts, zone analytics components, and compact benchmark-oriented outputs used in the study.

For a manuscript-ready version, see:

- [CODE_AVAILABILITY_STATEMENT.md](CODE_AVAILABILITY_STATEMENT.md)

## Data Availability

The full raw production video data are not publicly released in this repository. However, the repository includes code, compact derived outputs, benchmark summaries, prompts, and metadata sufficient to document the framework and support partial reproducibility.

For a manuscript-ready version, see:

- [DATA_AVAILABILITY_STATEMENT.md](DATA_AVAILABILITY_STATEMENT.md)

## Model Weights

Model weights are intentionally **not versioned in this public repository by default**.

This keeps the repository lighter and avoids bundling large binary assets into the initial public release. If weights are needed later, they can be:

- distributed separately,
- linked from a release page,
- or reproduced locally using the provided training scripts and notes.

## Citation

If you use this repository in academic work, please cite the associated manuscript when available.

You can also use the metadata in:

- [CITATION.cff](CITATION.cff)

## License

This repository is released under the MIT License.

See:

- [LICENSE](LICENSE)
