# Annotation Workflow

## Goal

Use `yolo11x.pt` to accelerate annotation for the article dataset while keeping the final benchmark clean.

## Rules

- `test` is manual-first and reviewed by hand.
- `hard_subset` is manual-first and selected for difficult cases only.
- Auto-labeling is allowed for `train` and `val` as a draft to be corrected.
- Every auto-labeled round should produce a report in `article_pack/data/cam10/seed_round1/reports/`.

## Round 1 Plan

- `train`: preview frames from `14:00` and `15:00`
- `val`: preview frames from `16:00`
- `test_candidates`: preview frames from `17:00`
- base auto-label model: `yolo11x.pt`

## Manual Workflow

Use the project virtual environment explicitly:

```powershell
venv\Scripts\python.exe article_pack\scripts\auto_label_review.py `
  --images article_pack\data\cam10\seed_round1\images\train `
  --labels article_pack\data\cam10\seed_round1\labels\train `
  --model yolo11x.pt `
  --conf 0.15 `
  --box-thickness 4 `
  --selected-thickness 7
```

For validation:

```powershell
venv\Scripts\python.exe article_pack\scripts\auto_label_review.py `
  --images article_pack\data\cam10\seed_round1\images\val `
  --labels article_pack\data\cam10\seed_round1\labels\val `
  --model yolo11x.pt `
  --conf 0.15 `
  --box-thickness 4 `
  --selected-thickness 7
```

If labels already exist and you only want to continue manual review:

```powershell
venv\Scripts\python.exe article_pack\scripts\auto_label_review.py `
  --images article_pack\data\cam10\seed_round1\images\train `
  --labels article_pack\data\cam10\seed_round1\labels\train `
  --model yolo11x.pt `
  --review `
  --box-thickness 4 `
  --selected-thickness 7
```

### Keys

- `Space` or `Enter`: save current labels and go next
- `Backspace`: go back
- `D`: delete selected bbox
- `C`: switch class of selected bbox
- `S`: select next bbox
- `0..9`: select bbox by index
- `N`: draw new `vat`
- `P`: draw new `person`
- `T`: quick fine-tune on reviewed frames, then re-predict remaining frames
- `Q` or `Esc`: save and quit

## Round 2 Expansion

Use the first trained detector for the next annotation round.

1. Extract fresh frames with a shifted sampling phase:

```powershell
venv\Scripts\python.exe article_pack\scripts\extract_round2_frames.py `
  --input-dir article_pack\data\cam10\source_video `
  --output-root article_pack\data\cam10\round2 `
  --count-train 30 `
  --count-val 30 `
  --phase 0.25
```

2. Auto-label and review `train` with the new seed model:

```powershell
venv\Scripts\python.exe article_pack\scripts\auto_label_review.py `
  --images article_pack\data\cam10\round2\images\train `
  --labels article_pack\data\cam10\round2\labels\train `
  --model runs\detect\article_pack\experiments\detector\cam10_seed_round1_yolo11x\weights\best.pt `
  --conf 0.20 `
  --box-thickness 4 `
  --selected-thickness 7
```

3. Then do the same for `val`:

```powershell
venv\Scripts\python.exe article_pack\scripts\auto_label_review.py `
  --images article_pack\data\cam10\round2\images\val `
  --labels article_pack\data\cam10\round2\labels\val `
  --model runs\detect\article_pack\experiments\detector\cam10_seed_round1_yolo11x\weights\best.pt `
  --conf 0.20 `
  --box-thickness 4 `
  --selected-thickness 7
```

## Article Framing

This step can be described as:

- seed annotation bootstrapping with a large pretrained detector
- manual correction of domain-specific labels
- iterative expansion of the training set after the first corrected round

## What To Save

- seed split manifest
- auto-label reports for `train` and `val`
- notes on typical correction patterns
- later: comparison of pretrained auto-label draft vs corrected labels
