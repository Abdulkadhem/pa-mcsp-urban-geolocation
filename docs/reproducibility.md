# Reproducibility record for the paper run

This document defines the exact primary configuration used for the manuscript.
It is deliberately separate from exploratory adapter, posterior-head, gate,
and reranker experiments.

## Primary method

- Run identifier: `patch_token_full_20260726`
- Backbone: `geolocal/StreetCLIP`, frozen
- Candidate support: 16 adaptive-S2 prototype candidates plus 16 global image-retrieval candidates
- Patch cue: mean cosine similarity over four quadrant-pooled ViT patch-token descriptors
- Calibration: row-wise z-score with epsilon `1e-6`
- Fixed fusion weight: `alpha=0.75`
- Nested development selections: `[0.75, 0.75, 0.50, 0.75, 0.75]`
- Trainable checkpoint used by the primary result: **none**

The headline official snapshot contains 38,190 urban queries. It reports
Acc@25 km `20.2304%`, Acc@200 km `38.2797%`, median error `432.017 km`,
mean error `1665.523 km`, and GeoScore `0.638854`.

## Evidence hashes

These hashes identify the local locked evidence pack from which the paper
tables were generated:

```text
official_patch_fusion_metrics.json    SHA256 6A7D40A56FCC11918E68EB39DB00E8831DF429CD41FDC34A02F8578B47C2DEC7
official_patch_fusion_bootstrap.json  SHA256 1D77D29CD6659298D8544CD7C4A7701A080DEB47A1F5120E8BE0B127437B52C6
patch_token_fusion_config.json        SHA256 C14E6F064374F98C4CD5FAB5BDC7EB5835C8C6E764BA6750D1BD6E4B42E19156
```

The files are not copied into the small code-and-weights release because the
prediction arrays and descriptor caches are paper artifacts. Keep the local
pack and these hashes together when preparing a submission archive.

## Required external inputs

Reproduction requires the licensed OSV-5M images and manifests, train/test
StreetCLIP feature arrays, the patch descriptor cache, and the exact image
paths used by the locked run. These inputs are intentionally not bundled in
GitHub. The scripts accept all paths explicitly and do not silently fall back
to local images, EXIF metadata, OCR, or test coordinates.

## Interpretation

The official split was inspected after development and is therefore reported
as a locked diagnostic snapshot. It should not be described as a clean
confirmatory holdout or as universal state-of-the-art evidence.
