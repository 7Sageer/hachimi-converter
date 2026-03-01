# Repository Guidelines

## Project Structure & Module Organization
`scripts/` contains the runnable pipeline and core modules:
- Data prep: `download.py`, `download_v2.py`, `align_v2.py`, `slice_v2.py`, `validate_pairs.py`
- Training/inference: `train.py`, `inference.py`, `visualize.py`
- Core model/audio logic: `model.py`, `hifigan_model.py`, `mel_utils.py`

`data/` stores raw and sliced audio (`original/`, `hachimi/`, `paired/`).  
`models/` stores trained U-Net checkpoints and HiFi-GAN weights.  
`output/` stores generated audio samples.  
`pairs.jsonl` and `pairs_full.jsonl` are dataset metadata sources.

## Build, Test, and Development Commands
Run from repository root with Python 3:
- `python scripts/download.py` - download paired audio from Bilibili metadata.
- `python scripts/download_v2.py` - fetch expanded pairs via API + YouTube.
- `python scripts/download_vocoder.py` - download HiFi-GAN weights.
- `python scripts/slice_v2.py` - align and slice training segments into `data/paired/`.
- `python scripts/train.py --epochs 120` - train U-Net and save checkpoints to `models/`.
- `python scripts/inference.py input.wav output.wav 15 0` - convert a 15s segment.
- `python scripts/validate_pairs.py` - score pair alignment quality.

## Coding Style & Naming Conventions
Use 4-space indentation and PEP 8 style. Prefer:
- `snake_case` for functions/files, `PascalCase` for classes, `UPPER_CASE` for constants.
- `pathlib.Path` for filesystem paths.
- Small, script-focused modules with clear CLI entry points.

Keep mel parameters consistent across scripts (especially values defined in `scripts/mel_utils.py`), as they are coupled to HiFi-GAN config.

## Testing Guidelines
There is no formal test suite yet. Before opening a PR:
- Run `python scripts/validate_pairs.py` after dataset updates.
- Run a small smoke workflow: short training run + one inference sample.
- Confirm artifacts are written as expected (`models/*.pt`, `output/*.wav`).

## Commit & Pull Request Guidelines
Follow existing commit style: short, imperative, lowercase summaries, often with scope (example: `add validate_pairs.py: chroma alignment quality check`).

PRs should include:
- What changed and why.
- Commands run and key results (loss trend, sample output path).
- Any data/model implications.

Do not commit generated artifacts from ignored paths (`data/`, `models/`, `__pycache__/`, `/tmp/`).
