# Open-Source Release Checklist

This checklist is for turning `article_pack/` into a clean standalone GitHub repository.

## Already In Good Shape

- framework-facing scripts are grouped inside `article_pack/scripts/`
- benchmark outputs and notes are already separated from older legacy material
- detector, tracker, and zone analytics summaries already exist
- figure prompts and article-oriented assets are already grouped

## Still Needs A Decision

- choose a final license
- decide whether model weights should be versioned, linked, or omitted
- decide whether all result assets should stay or only the summarized ones
- decide whether manuscript-specific prompts in `diploma/` should remain outside the public repo

## Recommended Publish Surface

Keep:

- `scripts/`
- `results/` with summarized CSV/MD and selected lightweight artifacts
- `tracking_benchmark/` metadata and GT files, but not large raw frame dumps
- `zones/`
- `notes/`
- `configs/`
- `observation_layer_prompt_files/`
- `README.md`
- `requirements.txt`

Do not publish by default:

- private or heavy source videos
- local zip archives
- full training dumps
- temporary frame-selection assets
- local environments

## Nice-To-Have Later

- move reusable runtime logic into a dedicated package such as `observation_layer/`
- add `pyproject.toml`
- add CI checks for linting and basic smoke tests
- add a small public demo dataset or synthetic example

## Working Recommendation

Do not do a disruptive refactor before the paper is stable.

The safer path is:

1. freeze the framework narrative
2. stabilize figures and paper assets
3. publish `article_pack` as the initial public repo core
4. refactor packaging only after the public scope is clear

