# PA-MCSP Urban Geolocation Toolkit

Reference implementation of the **PA-MCSP (Patch-Aware Multi-Cue Spatial Proposal)** urban image geolocation method.

Clean code-and-weights release for the urban image-geolocation system. This
repository intentionally excludes the manuscript, figures, experimental logs,
large image/feature datasets, cloud snapshots, and paper-specific evidence.

## Author

- Abdulkadhem A. Abdulkadhem — Al-Mustaqbal University
  - ORCID: https://orcid.org/0000-0002-4533-7083

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
bundled; pass their local paths explicitly to the scripts. The exact primary
paper run is documented in `docs/reproducibility.md` and
`configs/pa_mcsp_paper_locked.json`.

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

The primary paper result is the training-free patch-token late-fusion pipeline;
it does not load a trainable neural checkpoint. The `.pt` files in this release
are auxiliary checkpoints from exploratory variants and are not used to produce
the headline Table 3 result. Do not commit raw images, EXIF metadata, private
credentials, or generated result directories.

## Main entry points

```bash
python scripts/extract_streetclip_features_resumable.py --help
python scripts/run_official_patch_fusion.py --help
python scripts/evaluate_patch_fusion_nested.py --help
python scripts/validate_pa_mcsp_outputs.py --help
```

All paths are supplied as command-line arguments so the code can be used with
local storage, a mounted cloud volume, or a Slurm/Vast.ai workspace.

## Reproducibility scope

The paper reports a locked diagnostic snapshot rather than a confirmatory
holdout claim. The public repository contains the code and auxiliary weights;
the OSV-5M images, manifests, feature caches, and paper-specific prediction
arrays remain external because of their size and licensing constraints. See
`docs/reproducibility.md` for the exact method configuration, metric hashes,
and the required external inputs.
