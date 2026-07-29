# Checkpoint manifest

These files are the selected checkpoints copied from the validated project
workspace. They are model state only; no training images or derived feature
matrices are included.

Use Git LFS for checkpoint files when publishing the repository:

```bash
git lfs install
git lfs track "weights/*.pt"
```

The surrounding scripts record the expected checkpoint schema and will fail
with a clear error if a checkpoint from a different method is supplied.
