"""Manifest-driven subject-aware MRI dataset."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .target_generation import make_structural_targets
from .preprocessing import (
    paired_augment, read_image, read_mask, resize_pair, zscore_normalize,
)


REQUIRED_COLUMNS = {"subject_id", "image_path", "mask_path", "split"}


def load_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_path}")
    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return rows


def validate_subject_splits(rows: list[dict[str, str]]) -> None:
    """Raise if one subject appears in multiple split labels."""
    split_by_subject: dict[str, set[str]] = {}
    for row in rows:
        split_by_subject.setdefault(row["subject_id"], set()).add(row["split"])
    collisions = {s: splits for s, splits in split_by_subject.items() if len(splits) > 1}
    if collisions:
        example = list(collisions.items())[:5]
        raise ValueError(
            "Subject leakage detected: subject_id appears in multiple splits. "
            f"Examples: {example}"
        )


class KneeLigamentDataset(Dataset):
    """Load images and categorical ACL/PCL masks from a CSV manifest.

    Manifest columns:
      subject_id,image_path,mask_path,split
    Optional columns:
      spacing_y_mm,spacing_x_mm,slice_id
    Mask encoding is categorical: 0 background, 1 ACL, 2 PCL.
    """
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_root: str | Path = ".",
        image_size: int = 320,
        augment: bool = False,
        seed: int = 42,
        validate_all_splits: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.image_root = Path(image_root).resolve()
        rows = load_manifest(self.manifest_path)
        if validate_all_splits:
            validate_subject_splits(rows)
        self.rows = [r for r in rows if r["split"].strip().lower() == split.lower()]
        if not self.rows:
            raise ValueError(f"No records for split={split!r} in {manifest_path}")
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.split = split

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (self.image_root / path).resolve()

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = read_image(self._resolve(row["image_path"]))
        mask = read_mask(self._resolve(row["mask_path"]))
        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for subject {row['subject_id']}: "
                f"{image.shape} vs {mask.shape}"
            )
        image, mask = resize_pair(image, mask, (self.image_size, self.image_size))
        if self.augment:
            # Per-index deterministic seed, changed across epochs only if the
            # caller changes the dataset seed. For stochastic epoch-wise
            # augmentation, set seed in the training loop or extend this class.
            rng = np.random.default_rng(self.seed + index)
            image, mask = paired_augment(image, mask, rng)
        image = zscore_normalize(image)

        if not np.isin(mask, [0, 1, 2]).all():
            raise ValueError(
                f"Mask {row['mask_path']} must contain only labels 0, 1, 2."
            )
        one_hot = np.stack((mask == 1, mask == 2), axis=0).astype(np.float32)
        centerline, distance = make_structural_targets(
            mask, output_size=(self.image_size // 8, self.image_size // 8)
        )
        image_tensor = torch.from_numpy(image[None].astype(np.float32))
        mask_tensor = torch.from_numpy(one_hot)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "centerline": torch.from_numpy(centerline),
            "distance": torch.from_numpy(distance),
            "subject_id": row["subject_id"],
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "spacing_y_mm": float(row.get("spacing_y_mm") or "nan"),
            "spacing_x_mm": float(row.get("spacing_x_mm") or "nan"),
            "slice_id": row.get("slice_id", str(index)),
        }
