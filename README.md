# ARS-Net 

This repository contains a PyTorch implementation of the ARS-Net architecture, including its multi-scale encoder, geometry-guided relational module, structural refinement, progressive decoder, auxiliary structural targets, composite loss, subject-level data splitting, training, inference, physical-distance evaluation, ablations, statistics.


## 1. Environment

Use Python 3.9 and PyTorch 2.0.1 environment. Install the appropriate PyTorch wheel for the machine's CUDA driver first, then install the remaining dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

Check the installation with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

## 2. Exact local dataset paths and format

The project uses paths relative to the repository root; no developer-specific  paths are embedded in scripts.

Place authorized, de-identified data in this structure:

```text
data/private_dataset/
├── images/<SUBJECT_ID>/<SLICE_ID>.png
├── masks/<SUBJECT_ID>/<SLICE_ID>.png
└── metadata.csv
```

The image and mask relative paths must match. 
Build the all-slices manifest from the repository root:

```bash
python scripts/build_manifest.py --dataset-root data/private_dataset --output manifests/all_slices.csv
```

The script checks image/mask correspondence, metadata coverage, duplicate subject/slice keys, and positive physical spacing. It writes image paths as `images/...` and mask paths as `masks/...`; therefore all training/evaluation commands use `--image-root data/private_dataset`.

## 3. Create fixed subject-level CV manifests

We used 200 subjects, with 40 independent test subjects and 160 development subjects. The script assigns subjects, not individual slices, to the split and writes five fixed manifests. Use the same manifests for ARS-Net and every baseline.

```bash
python scripts/make_cv_manifests.py \
  --input-csv manifests/all_slices.csv \
  --output-dir manifests/cv_5fold_160_40 \
  --test-fraction 0.2 --folds 5 --seed 42
```

Inspect the resulting files before training. Preserve them and record checksums. The 40 test subjects must remain untouched during model selection. Do not regenerate the split independently for different methods.

## 4. Train ARS-Net

Run one fold (fold 1 shown):

```bash
python scripts/train.py \
  --manifest manifests/cv_5fold_160_40/fold1.csv \
  --image-root data/private_dataset \
  --output-dir outputs/ars_net/fold1 \
  --epochs 150 --batch-size 4 --lr 0.0002 --weight-decay 0.0001 \
  --warmup-epochs 5 --seed 42 --amp
```

The training script writes `best.pt`, `last.pt`, and `run_config.json`. It uses AdamW, five-epoch linear warm-up followed by cosine decay, gradient clipping at 1.0, and optional mixed precision. The default composite objective is `L4`; `--architecture-variant` selects `full`, `no_arrm`, `no_msfm`, or `no_arrm_no_msfm`, while `--loss-variant` selects `L1` through `L4` as documented in `docs/ablation_studies.md`.

Run the full ablation suite for one fold using the same manifest:

```bash
python scripts/run_ablation_suite.py \
  --manifest manifests/cv_5fold_160_40/fold1.csv \
  --image-root data/private_dataset \
  --output-root outputs/ablations/fold1 \
  --epochs 150 --batch-size 4 --amp --suite all
```

Run this separately for each fold if you need fold-level ablation results. The suite launches each variant as a separate process and stops if any run fails.

## 5. Evaluate ARS-Net and compute physical distances

```bash
python scripts/evaluate.py \
  --manifest manifests/cv_5fold_160_40/fold1.csv \
  --image-root data/private_dataset \
  --checkpoint outputs/ars_net/fold1/best.pt \
  --split test \
  --output-csv outputs/evaluation/fold1_test_slice_metrics.csv

python scripts/aggregate_subject_metrics.py \
  --input-csv outputs/evaluation/fold1_test_slice_metrics.csv \
  --output-subject-csv outputs/evaluation/fold1_test_subject_metrics.csv \
  --output-summary-csv outputs/evaluation/fold1_test_summary.csv \
  --bootstrap 10000 --seed 42
```

Predictions are resized back to each original mask grid before overlap and surface-distance metrics are calculated. Aggregate slices within subject before confidence intervals and paired tests.

For inference on a single image:

```bash
python scripts/inference.py --image data/private_dataset/images/SUBJ001/000.png \
  --checkpoint outputs/ars_net/fold1/best.pt \
  --output outputs/predictions/SUBJ001_000_mask.png
```

## 6. Baseline status and fair comparison


The official repositories for TransUNet, Swin-Unet, BATFormer, UCTNet, nnU-Net, Switch-UMamba, and ACM-UNet are listed with their exact clone URLs and local paths in `configs/baselines/external_sources.json`. Clone them with:

```bash
python scripts/setup_external_baselines.py
```

Or clone selected repositories, for example:

```bash
python scripts/setup_external_baselines.py --only ACM-UNet,BATFormer,TransUNet
```

** Details are in [`docs/sota_comparison.md`](docs/sota_comparison.md). The generic baseline runner accepts a factory that returns a PyTorch model taking `[B,1,320,320]` and outputting two-channel logits `[B,2,320,320]`.

Run the local custom baseline over the same folds with:

```bash
python scripts/run_sota_cv.py \
  --registry configs/baselines/registry.json \
  --manifest-template 'manifests/cv_5fold_160_40/fold{fold}.csv' \
  --image-root data/private_dataset \
  --output-root outputs/baselines/afc_unet_custom \
  --epochs 150 --batch-size 4 --amp
```

## 7. Tests and audit

From the repository root:

```bash
python -m pytest -q
python scripts/measure_efficiency.py --help
python scripts/build_manifest.py --help
python scripts/run_ablation_suite.py --help
python scripts/run_sota_cv.py --help
```


## 8. Repository map

- `models/`: ARS-Net encoder, geometry-guided relational module, structural feedback/refinement, decoder, and model wrapper.
- `datasets/`: image/mask loading, augmentation, normalization, and centerline/distance target generation.
- `losses/`: segmentation, centerline, distance, and composite objectives.
- `scripts/`: manifest creation, splitting, training, inference, evaluation, ablations, baseline training, statistical aggregation, visualization, and efficiency measurement.
- `evaluation/`: overlap metrics, physical surface distances, statistical tests, and efficiency utilities.
- `baselines/`: factory interface for external models.
- `configs/`: concrete data configuration, runnable local baseline registry, and official external repository URLs.
- `results/*reference.csv`: Results generated.
- `docs/`: dataset access, reproducibility, ablation, baseline integration, and release audit instructions.
