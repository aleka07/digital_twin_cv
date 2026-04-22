# Dataset Manifest

## Primary Scene

- Camera: `CAM10`
- Role: main article dataset
- Source date: `2026-03-23`
- Source format: `4` hourly MP4 files
- Resolution / FPS: `1280x1920 @ 6 FPS`

## Secondary Scene

- Camera: `CAM24`
- Role: robustness / future-work support

## CAM10 Source Inventory

| File | Covered hour | Duration | Notes |
| --- | --- | --- | --- |
| `1_10_R_23032026140000.mp4` | `14:00-14:59` | `~60 min` | candidate train pool |
| `1_10_R_23032026150000.mp4` | `15:00-15:59` | `~60 min` | candidate train pool |
| `1_10_R_23032026160000.mp4` | `16:00-16:59` | `~60 min` | candidate val pool |
| `1_10_R_23032026170000.mp4` | `17:00-17:59` | `~60 min` | fixed test / hard-subset / tracking pool |

## Split Policy

We use time-based separation instead of random frame-level mixing.

- `train` candidates: mostly `14:00` and `15:00`
- `val` candidates: mostly `16:00`
- `test` candidates: only `17:00`
- `hard_subset`: selected from `17:00` only
- `tracking clips`: selected from `17:00` first, unless diversity forces one clip from `16:00`

This policy is meant to reduce leakage from near-duplicate neighboring frames across splits.

## Planned Assets

| Asset | Target | Status | Notes |
| --- | --- | --- | --- |
| CAM10 preview scouting set | 120 frames | prepared | `30` evenly sampled frames from each hourly video |
| Seed round 1 train | 60 preview frames | manually reviewed | `14:00` + `15:00`, all `60` label files present |
| Seed round 1 val | 30 preview frames | manually reviewed | `16:00`, all `30` label files present |
| Round 2 train | 60 extracted frames | auto-labeled | shifted sampling (`phase=0.25`) from `14:00` + `15:00`, labeled with seed `best.pt` |
| Round 2 val | 30 extracted frames | auto-labeled | shifted sampling (`phase=0.25`) from `16:00`, labeled with seed `best.pt` |
| Seed round 1 test candidates | 30 preview frames | prepared | `17:00`, reserved for manual review only |
| Detection test set draft | 150 extracted frames | prepared | `17:00`, manual-first benchmark set |
| Hard subset pool | 60 extracted frames | prepared | `17:00`, manual-first difficult-case pool |
| Detection train split | prepared | ready for first training | `60` images, `198` boxes |
| Detection val split | prepared | ready for first training | `30` images, `116` boxes |
| Detection test split | 120-150 frames | pending |  |
| Hard subset | 40-60 frames | pending |  |
| Tracking clip A | 180 frames | prepared | `17:10:00-17:10:29`, ready for ID annotation |
| Tracking clip B | 180 frames | prepared | `17:42:00-17:42:29`, ready for ID annotation |
| CAM24 robustness frames | 30-50 frames | pending |  |
