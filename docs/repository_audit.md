# Repository verification and release audit

## Implemented in this repository

- ARS-Net model code: multi-scale encoder, geometry-guided relational module, structural feature refinement, and progressive decoder.
- Dataset loading, paired augmentation, normalization, structural target generation, subject-level manifest validation, and fixed 160/40 split plus five-fold manifests.
- Composite segmentation/centerline/distance loss and L1-L4 loss variants.
- Training, checkpointing, inference, restored-grid overlap/physical-distance evaluation, subject aggregation, bootstrap confidence intervals, paired tests with Holm correction, visualization, and efficiency measurement.
- Architecture/loss ablation studies.
- Exact upstream URLs and fixed local clone destinations for the named external baseline repositories.

## Release checklist

- [ ] Run `python -m pytest -q` in the target environment.
- [ ] Run `python scripts/check_baseline_registry.py`.
- [ ] Validate manifest paths, label values, spacing, subject counts, and split isolation.
- [ ] Train all intended folds and ablations.
- [ ] Evaluate the independent test set only after all model-selection decisions are frozen.
- [ ] Recompute metrics and tables from generated predictions, aggregating slices at subject level.

