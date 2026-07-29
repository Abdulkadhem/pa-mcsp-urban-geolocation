# PA-MCSP Urban Geolocation Toolkit

Reference implementation of the **PA-MCSP (Prototype-Adaptive Multi-Cue Spatial Prior)** urban image geolocation method.

Clean code-and-weights release for the urban image-geolocation system. This
repository intentionally excludes the manuscript, figures, experimental logs,
large image/feature datasets, cloud snapshots, and paper-specific evidence.

## Authors

- Abdulkadhem A. Abdulkadhem — Al-Mustaqbal University
  - ORCID: https://orcid.org/0000-0002-4533-7083
- Zain I. Al-Tamimi — Al-Mustaqbal University
  - ORCID: https://orcid.org/0009-0005-0848-6855

## Repository layout

```text
src/          reusable spherical-geolocation and PA-MCSP modules
scripts/      feature extraction, training, evaluation, and validation entry points
configs/      JSON experiment configurations
tests/        lightweight unit tests
weights/      selected trained checkpoints
```

## Installation

Python 3.10 or newer is recommended. For a CUDA environment, install a
PyTorch build matching the installed CUDA toolkit first, then install the
remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
```

The public StreetCLIP backbone is downloaded by the extraction scripts from
its Hugging Face model identifier. Dataset files and feature caches are not
bundled; pass their local paths explicitly to the scripts.

## Checkpoints

The `weights/` directory contains only small model checkpoints. Large feature
matrices and candidate caches are deliberately excluded because they are
derived data rather than model weights and are not suitable for a normal
GitHub repository.

| File | Role |
|---|---|
| `streetclip_spatial_adapter.pt` | residual adapter for StreetCLIP features |
| `urban_s2_head_v2_full.pt` | spherical urban posterior head |
| `pa_mcsp_reranker_k16.pt` | candidate reranker for the 16-per-source setting |
| `pa_mcsp_binary_gate.pt` | binary candidate-selection gate |
| `pa_mcsp_ood_gate.pt` | out-of-distribution gate |

The checkpoints are loaded with `torch.load(..., map_location=...)`. Do not
commit raw images, EXIF metadata, private credentials, or generated result
directories.

## Main entry points

```bash
python scripts/extract_streetclip_features_resumable.py --help
python scripts/run_official_patch_fusion.py --help
python scripts/evaluate_patch_fusion_nested.py --help
python scripts/validate_pa_mcsp_outputs.py --help
```

All paths are supplied as command-line arguments so the code can be used with
local storage, a mounted cloud volume, or a Slurm/Vast.ai workspace.

## Publish to GitHub

From this directory:

```bash
git init -b main
git add .
git commit -m "Initial code and checkpoint release"
git remote add origin https://github.com/<account>/<repository>.git
git push -u origin main
```

If checkpoint files are retained, enable Git LFS before committing them.
