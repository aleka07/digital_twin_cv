# Tracking Workflow

## Goal

Prepare two short held-out tracking clips from `CAM10 17:00` and annotate stable `track_id` values for `person` and `vat`.

## Clip Policy

- `clip_A`: working scene with motion around `17:10`
- `clip_B`: denser / harder scene around `17:42`
- Both clips stay inside the held-out `17:00` hour
- Detector is fixed: `YOLO11x + freeze=22`

## Files

- Frames:
  - `article_pack/tracking_benchmark/clip_A/frames/`
  - `article_pack/tracking_benchmark/clip_B/frames/`
- Ground truth to edit:
  - `article_pack/tracking_benchmark/clip_A/gt_tracks.txt`
  - `article_pack/tracking_benchmark/clip_B/gt_tracks.txt`
- Draft helper files:
  - `draft_bytetrack.txt`
  - `draft_botsort.txt`
- Per-frame mapping:
  - `frame_manifest.csv`

## GT Format

Header:

```text
# frame_id,track_id,class_id,x1,y1,x2,y2
```

Example:

```text
1,1,0,182,1314,312,1918
1,2,1,826,1239,984,1628
2,1,0,180,1312,314,1918
```

Where:

- `frame_id`: starts from `1` inside each clip
- `track_id`: stable identity across frames inside one clip
- `class_id`: `0=person`, `1=vat`
- `x1,y1,x2,y2`: integer bbox corners

## Annotation Rules

- One physical object keeps one `track_id` while visible.
- If the object disappears for a short occlusion and clearly returns, keep the same `track_id`.
- If you are no longer confident it is the same object, start a new `track_id`.
- IDs are local to each clip. `clip_A` and `clip_B` do not need shared identities.
- Use visible object extent, same bbox logic as detector annotation.
- Do not annotate shadows or reflections.

## Practical User Workflow

Launch for `clip_A`:

```powershell
venv\Scripts\python.exe article_pack\scripts\review_tracking_gt.py `
  --clip-dir article_pack\tracking_benchmark\clip_A
```

Launch for `clip_B`:

```powershell
venv\Scripts\python.exe article_pack\scripts\review_tracking_gt.py `
  --clip-dir article_pack\tracking_benchmark\clip_B
```

1. Open `frame_manifest.csv` to understand frame order and wall-clock time.
2. Use `draft_bytetrack.txt` as the starting point if `gt_tracks.txt` is still empty.
3. The tool saves corrected lines directly into `gt_tracks.txt`.
4. Keep IDs simple and consecutive when possible.
5. When a draft track is wrong, fix the ID instead of preserving the tracker's mistake.
6. If ByteTrack misses an object, add the missing boxes directly in the tool.

## Keys

- `Space` / `Enter` - save current frame and go next
- `Backspace` - save current frame and go previous
- `S` - select next box
- `0..9` - select box by index
- `D` - delete selected box
- `I` - change `track_id` of selected box inside the HUD overlay
- `C` - toggle class of selected box
- `N` - draw new `vat` box and assign a `track_id` inside the HUD overlay
- `P` - draw new `person` box and assign a `track_id` inside the HUD overlay
- `M` - move selected box with the mouse, confirm with `Enter` or `Space`
- `R` - accept suggested `track_id` from the previous frame
- `A` - add the first object that looks missing from the previous frame
- `Q` / `Esc` - save all progress and quit

## Expected Output

Each clip should end with:

- non-empty `gt_tracks.txt`
- stable IDs for all main moving `person` and `vat` objects
- enough consistency to run `ByteTrack vs BoT-SORT` metrics afterward
