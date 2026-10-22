# SOTA comparison: exact source locations and run status


## Official upstream source locations
`configs/baselines/registry.json` enables `AFC-Unet`, built from `baselines/afc_unet.py` and loaded through `baselines.afc_factory:build_model`. 

The exact URLs and local clone destinations are stored in `configs/baselines/external_sources.json`. Run `python scripts/setup_external_baselines.py` from the repository root to clone the configured repositories under `external/`. 

| Method | Local source path after setup | 
|---|---|
| TransUNet | `external/TransUNet` | 
| Swin-Unet | `external/Swin-Unet` | 
| BATFormer | `external/BATFormer` | 
| UCTNet | `external/UCTNet` | 
| nnU-Net | `external/nnUNet` | 
| Switch-UMamba | `external/Switch-UMamba` | 
| ACM-UNet | `external/ACM-UNet` | 

## Common adapter contract

For models trained through `scripts/train_baseline.py`, a factory must return `torch.nn.Module` accepting grayscale images `[B,1,320,320]` and producing raw logits `[B,2,320,320]` in channel order ACL then PCL. `baselines/adapter.py` normalizes tensor, tuple/list, or supported dictionary outputs and validates the two-channel output. If a model returns multiple auxiliary outputs, the factory/adapter must explicitly select its main segmentation output. 



## Run the currently enabled local baseline

```bash
python scripts/run_sota_cv.py \
  --registry configs/baselines/registry.json \
  --manifest-template 'manifests/cv_5fold_160_40/fold{fold}.csv' \
  --image-root data/private_dataset \
  --output-root outputs/baselines/afc_unet_custom \
  --epochs 150 --batch-size 4 --amp
```


## Metrics and statistical analysis

Use the same fixed subject-level manifests for all methods. Restore predicted masks to the original mask grid before calculating ASD/HD95 with the original DICOM pixel spacing. Aggregate the ten slices per subject before bootstrap confidence intervals and paired tests. The independent 40-subject test set must not be used for model selection. `results/table*_reference.csv` files are the results generated from the methods.

## Upstream references

- TransUNet: https://github.com/Beckschen/TransUNet
- Swin-Unet: https://github.com/HuCaoFighting/Swin-Unet
- BATFormer: https://github.com/xianlin7/BATFormer
- UCTNet: https://github.com/innocence0206/UCTNet
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet
- Switch-UMamba: https://github.com/Maqkccx/Switch-UMamba
- ACM-UNet: https://github.com/zyklcode/ACM-UNet
