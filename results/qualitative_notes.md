# Qualitative Notes

Use this file to collect:
- best examples
- failure cases
- domain gap observations
- tracker behavior notes
- figure candidates for the paper

## Early Autolabel Observations

- `yolo11x.pt` bootstrap labels are useful as a draft, but coverage is sparse on `CAM10`.
- Round 1 counts:
  - `train`: `60` images, `54` detections, `35` non-empty frames
  - `val`: `30` images, `14` detections, `9` non-empty frames
- Likely issues to watch during manual correction:
  - missed `vat` detections due to domain mismatch
  - false positive COCO-to-`vat` mappings
  - empty frames that should remain explicitly empty

## Post-Review Seed Status

- User completed full manual review for `seed_round1/train` and `seed_round1/val`.
- Final cleaned seed set:
  - `train`: `60` images, `198` boxes
  - `val`: `30` images, `116` boxes
- This is now the baseline dataset for the first detector fine-tune.

## First Seed Training Result

- First domain-adapted detector trained from `yolo11x.pt` on `CAM10 seed_round1`.
- Best validation metrics:
  - `Precision`: `0.852`
  - `Recall`: `0.816`
  - `mAP50`: `0.856`
  - `mAP50-95`: `0.413`
- Per-class snapshot:
  - `person`: `mAP50=0.760`, `mAP50-95=0.316`
  - `vat`: `mAP50=0.952`, `mAP50-95=0.510`
- Interpretation:
  - `vat` adapts faster on the current seed set
  - `person` still needs more diverse examples and likely more negatives / occlusions

## Round 2 Bootstrap Quality

- `round2/train`: `60` images, `252` draft detections from the seed model
- `round2/val`: `30` images, `125` draft detections from the seed model
- Compared with the initial COCO bootstrap, the domain-adapted model now produces dense labels on nearly all new frames.
- Expected manual focus in round 2:
  - bbox tightening
  - false positives on background structures
  - harder `person` cases
  - preserving some true-empty frames later in dedicated negative sampling

## Round12 Training Result

- Second detector trained on merged `seed_round1 + round2`.
- Best validation metrics:
  - `Precision`: `0.913`
  - `Recall`: `0.900`
  - `mAP50`: `0.927`
  - `mAP50-95`: `0.591`
- Per-class snapshot:
  - `person`: `mAP50=0.876`, `mAP50-95=0.444`
  - `vat`: `mAP50=0.978`, `mAP50-95=0.739`
- Improvement versus seed-only model:
  - `mAP50`: `0.856 -> 0.927`
  - `mAP50-95`: `0.413 -> 0.591`
- Interpretation:
  - dataset scaling already helps substantially
  - `person` remains the weaker class and still deserves targeted hard examples

## Held-Out 17:00 Evaluation

- These `17:00` frames were kept out of training and used as the first real held-out evaluation slice.
- Clean test (`150` images, `593` boxes):
  - `Precision`: `0.883`
  - `Recall`: `0.884`
  - `mAP50`: `0.939`
  - `mAP50-95`: `0.754`
- Hard-pool (`60` images, `231` boxes):
  - `Precision`: `0.893`
  - `Recall`: `0.894`
  - `mAP50`: `0.936`
  - `mAP50-95`: `0.821`
- Per-class on clean test:
  - `person`: `P=0.809`, `R=0.846`, `mAP50=0.891`, `mAP50-95=0.658`
  - `vat`: `P=0.956`, `R=0.922`, `mAP50=0.988`, `mAP50-95=0.851`
- Interpretation:
  - the model generalizes well to new positions and scene states in the held-out hour
  - `person` is still the weaker class and remains the main target for future data expansion

## Fine-Tuning Depth Ablation

- Controlled comparison on the same `merged_round12` train/val split and the same fixed `17:00` held-out evaluation sets:
  - `freeze=22`:
    - val: `P=0.913`, `R=0.900`, `mAP50=0.927`, `mAP50-95=0.591`
    - clean test: `P=0.883`, `R=0.884`, `mAP50=0.939`, `mAP50-95=0.754`
    - hard-pool: `P=0.893`, `R=0.894`, `mAP50=0.936`, `mAP50-95=0.821`
  - `freeze=10`:
    - val: `P=0.958`, `R=0.887`, `mAP50=0.954`, `mAP50-95=0.600`
    - clean test: `P=0.894`, `R=0.835`, `mAP50=0.894`, `mAP50-95=0.603`
    - hard-pool: `P=0.918`, `R=0.861`, `mAP50=0.906`, `mAP50-95=0.628`
  - `freeze=0`:
    - val: `P=0.888`, `R=0.871`, `mAP50=0.909`, `mAP50-95=0.495`
    - clean test: `P=0.827`, `R=0.766`, `mAP50=0.836`, `mAP50-95=0.523`
    - hard-pool: `P=0.769`, `R=0.761`, `mAP50=0.791`, `mAP50-95=0.540`
- Interpretation:
  - deeper unfreezing helps or preserves the in-split validation score, but harms true held-out generalization
  - `freeze=22` remains the strongest recipe on both `17:00` evaluation slices
  - `freeze=10` is still useful as an ablation because it shows mild overadaptation even when val looks attractive
  - `freeze=0` clearly overfits this dataset scale and is not the right recipe for the article benchmark

## Model Size Sweep

- Controlled detector-size comparison with the fixed recipe `freeze=22` on `merged_round12`.
- Validation snapshot:
  - `YOLO11n`: `P=0.932`, `R=0.869`, `mAP50=0.913`, `mAP50-95=0.526`
  - `YOLO11s`: `P=0.892`, `R=0.921`, `mAP50=0.925`, `mAP50-95=0.540`
  - `YOLO11m`: `P=0.918`, `R=0.838`, `mAP50=0.902`, `mAP50-95=0.544`
  - `YOLO11l`: `P=0.899`, `R=0.865`, `mAP50=0.912`, `mAP50-95=0.551`
  - `YOLO11x`: `P=0.913`, `R=0.900`, `mAP50=0.927`, `mAP50-95=0.591`
- Held-out clean test (`17:00`):
  - `YOLO11n`: `mAP50=0.869`, `mAP50-95=0.559`, `2.2 ms/img`
  - `YOLO11s`: `mAP50=0.901`, `mAP50-95=0.603`, `4.3 ms/img`
  - `YOLO11m`: `mAP50=0.900`, `mAP50-95=0.598`, `9.4 ms/img`
  - `YOLO11l`: `mAP50=0.928`, `mAP50-95=0.645`, `11.9 ms/img`
  - `YOLO11x`: `mAP50=0.939`, `mAP50-95=0.754`, `18.2 ms/img`
- Held-out hard-pool:
  - `YOLO11n`: `mAP50=0.860`, `mAP50-95=0.571`, `2.0 ms/img`
  - `YOLO11s`: `mAP50=0.912`, `mAP50-95=0.667`, `4.2 ms/img`
  - `YOLO11m`: `mAP50=0.927`, `mAP50-95=0.657`, `9.3 ms/img`
  - `YOLO11l`: `mAP50=0.927`, `mAP50-95=0.697`, `11.7 ms/img`
  - `YOLO11x`: `mAP50=0.936`, `mAP50-95=0.821`, `18.4 ms/img`
- Interpretation:
  - `YOLO11x` remains the strongest detector on both held-out evaluation slices
  - `YOLO11l` is the strongest lighter alternative and gives the best non-`x` tradeoff
  - `YOLO11s` looks especially attractive if inference speed matters more than absolute best quality
  - `vat` is learned well across all model sizes, while `person` remains the limiting class on smaller models

## Tracking Benchmark

- Final GT clips completed:
  - `clip_A`: `180` frames, `1054` labeled instances, `7` unique track IDs
  - `clip_B`: `180` frames, `899` labeled instances, `5` unique track IDs
- Combined benchmark on `clip_A + clip_B` with detector `YOLO11x`:
  - `ByteTrack`: `HOTA=0.783`, `MOTA=0.718`, `IDF1=0.787`, `IDSW=15`, `FPS=27.6`
  - `BoT-SORT`: `HOTA=0.767`, `MOTA=0.719`, `IDF1=0.786`, `IDSW=17`, `FPS=17.0`
- Per-clip pattern:
  - `clip_A` is the harder sequence and causes all observed ID switches in both trackers
  - `clip_B` is noticeably easier; both trackers reached `IDSW=0` there
- Interpretation:
  - `ByteTrack` is the better main tracking baseline for this article package because it gives slightly stronger `HOTA/IDF1` and is substantially faster
  - `BoT-SORT` does not win in the current default Ultralytics setup, likely because the present configuration runs without dedicated ReID
  - the tracking bottleneck is concentrated in harder occlusion-heavy interaction segments rather than in the easier production flow segments

## Tracking Follow-Up Attempt

- A stronger follow-up variant was attempted with `BoT-SORT + ReID`.
- Two practical outcomes:
  - `model=auto` hit an upstream bug in the installed `ultralytics` version when trying to use native detector features for ReID
  - fallback to a classification-model ReID path required `yolo11n-cls.pt`, but downloading that extra weight failed in the current network environment
- Interpretation:
  - the current article benchmark should stay anchored on the completed `ByteTrack vs BoT-SORT` comparison
  - if stronger real-world tracking is still needed later, the next sensible step is either a fixed Ultralytics version with working ReID or a move to `BoxMOT / Deep-OC-SORT`

## Zone Analytics

- The first zone pass was run on user-finalized polygons from `article_pack/zones/cam10_zones_final.json`.
- The current five-zone layout is serviceable for exploration, but the analytics are not equally informative across zones.
- Main observations:
  - `front_vat_staging` is the most informative and stable zone across both clips
  - `right_loading_area` is also useful and shows consistent occupancy/dwell
  - `machine_service_area` works, but is more sensitive to polygon geometry and track center placement
  - `rack_buffer_area` is scene-specific and contributes mainly in `clip_A`
  - `left_transit_lane` is nearly inactive with the current `bbox center` membership rule
- Interpretation:
  - for the article narrative, a simplified `3-zone` subset would likely read more cleanly than the full `5-zone` exploratory layout
  - if we later want stronger human-zone analytics, the best next algorithmic upgrade is `footpoint` membership for `person` while keeping center-based membership for `vat`

## Long Zone Demo

- A full-hour demonstrational zone run was completed on `CAM10 17:00` using `YOLO11x + ByteTrack`.
- Main outputs live under `article_pack/results/zone_demo/cam10_1700_fullhour_bytetrack/`.
- Strongest long-run zones:
  - `machine_service_area`: `avg_occ=1.364`, `occupied_pct=95.18`, `total_dwell_s=4911.838`
  - `front_vat_staging`: `avg_occ=1.111`, `occupied_pct=68.9`, `total_dwell_s=4001.334`
- Secondary zones:
  - `rack_buffer_area` remains useful as a buffer/queue signal
  - `right_loading_area` is active but lighter
  - `left_transit_lane` is still weak and not a strong article zone in the current rule set
- Important limitation exposed by the long run:
  - the tracker produced `639` unique track IDs over one hour, which is far above the real object count and indicates substantial long-horizon ID fragmentation
  - because of that, long-run dwell and entry/exit aggregates are still useful as operational demo signals, but they should not be overclaimed as identity-stable analytics across the entire hour
