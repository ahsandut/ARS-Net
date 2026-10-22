# ARS-Net ablation studies

The repository includes executable architecture and objective ablations. 

## Architecture ablations

| `--architecture-variant` | Configuration |
|---|---|
| `full` | Full ARS-Net with ARRM and MSFM |
| `no_arrm` | Removes ARRM; projects E4 to the relational width before MSFM |
| `no_msfm` | Removes MSFM structural prediction/feedback; fuses the relational feature and E3 directly |
| `no_arrm_no_msfm` | Removes both ARRM and MSFM; uses E4 projection and direct E3 fusion |



## Loss ablations

| `--loss-variant` | Objective |
|---|---|
| `L1` | Segmentation BCE + Dice |
| `L2` | L1 + distance-map loss, weight 0.2 |
| `L3` | L1 + centerline loss, weight 0.3 |
| `L4` | L1 + centerline loss (0.3) + distance-map loss (0.2) |


## Run one experiment

```bash
python scripts/train.py \
  --manifest manifests/cv_5fold_160_40/fold1.csv \
  --image-root data/private_dataset \
  --architecture-variant no_arrm \
  --loss-variant L4 \
  --output-dir outputs/fold1/no_arrm_L4 \
  --epochs 150 --batch-size 4 --seed 42
```

## Run a suite

```bash
# Four architecture configurations, all using L4
python scripts/run_ablation_suite.py --manifest manifests/cv_5fold_160_40/fold1.csv \
  --image-root data/private_dataset --suite architecture --output-root outputs/fold1/architecture

# Four loss configurations, using the full architecture
python scripts/run_ablation_suite.py --manifest manifests/cv_5fold_160_40/fold1.csv \
  --image-root data/private_dataset --suite loss --output-root outputs/fold1/loss

# All unique architecture/loss runs
python scripts/run_ablation_suite.py --manifest manifests/cv_5fold_160_40/fold1.csv \
  --image-root data/private_dataset --suite all --output-root outputs/fold1/all
```

The `all` suite runs the architecture comparison with L4 and the full model's
L1-L4 loss comparison. It does not run every possible Cartesian combination.
For five-fold cross-validation, prepare a separate manifest for each fold and
run the suite separately per fold, using identical fold definitions for every
variant. Keep the independent test subjects out of training and model
selection.
