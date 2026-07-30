# Checkpoint manifest

These files are selected checkpoints copied from the validated project
workspace. They are model state only; no training images or derived feature
matrices are included.

Important scope note: the primary manuscript result is the learned-free
StreetCLIP patch-token late-fusion run described in
`configs/pa_mcsp_paper_locked.json`; it does **not** load a trainable
checkpoint. The files below belong to auxiliary exploratory variants and are
released for code continuity, not as inputs to the headline Table 3 result.

Use Git LFS for checkpoint files when publishing the repository:

```bash
git lfs install
git lfs track "weights/*.pt"
```

The surrounding scripts record the expected checkpoint schema and will fail
with a clear error if a checkpoint from a different method is supplied. Do not
interpret the presence of these files as evidence that the primary paper run
used them.
