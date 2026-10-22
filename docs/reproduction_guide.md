# Reproduction guide

This repository contains ARS-Net implementation and training/evaluation pipeline. 
## Dataset and manifest

Use `python scripts/build_manifest.py --dataset-root data/private_dataset --output manifests/all_slices.csv`. The dataset directory layout and required `metadata.csv` columns are defined in `data/README.md`. The generated manifest has `subject_id,image_path,mask_path,spacing_y_mm,spacing_x_mm,slice_id,split` columns. Run `scripts/make_cv_manifests.py` to create the five subject-level fold manifests and a shared independent test split.

The split field must be `train`, `val`, or `test` in fold manifests. All slices from one subject must remain in the same split. Mask values are 0=background, 1=ACL, 2=PCL.

## Environment

- Python 3.9
- PyTorch 2.0.1 (install the wheel matching the local CPU/CUDA setup)
- NumPy, OpenCV, SciPy, scikit-image, matplotlib, pytest


## Training and evaluation

Use the complete commands in the root README. The training pipeline creates centerline targets by skeletonizing downsampled masks and distance targets using normalized interior Euclidean distance transforms.

Predictions are restored to the original mask dimensions before overlap and surface-distance metrics are calculated. ASD/HD95 use original DICOM row/column pixel spacing. Aggregate slice-level measurements to subject-level summaries before bootstrap confidence intervals and paired statistical tests. Do not use the 40 independent test subjects for model selection.
