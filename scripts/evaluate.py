"""Evaluate a checkpoint on a manifest split. Predictions are restored to the original mask grid before overlap and physical-distance metrics are computed."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from datasets.dataset import KneeLigamentDataset
from models import ARSNet
from evaluation.metrics import segmentation_metrics
from evaluation.physical_distances import surface_distances_mm
from datasets.preprocessing import read_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output-csv", default="evaluation.csv")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    image_size = int(ckpt.get("args", {}).get("image_size", 320)) if isinstance(ckpt, dict) else 320
    ds = KneeLigamentDataset(args.manifest, args.split, args.image_root, image_size=image_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    architecture_variant = ckpt.get("args", {}).get("architecture_variant", "full") if isinstance(ckpt, dict) else "full"
    model = ARSNet(input_size=image_size, architecture_variant=architecture_variant).to(device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    rows = []
    image_root = Path(args.image_root).resolve()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            probs = torch.sigmoid(out["segmentation_logits"]).cpu().numpy()
            for i in range(probs.shape[0]):
                mask_path = Path(batch["mask_path"][i])
                if not mask_path.is_absolute():
                    mask_path = image_root / mask_path
                original_mask = read_mask(mask_path)
                original_h, original_w = original_mask.shape
                # Restore probabilities to the original mask grid before both
                # overlap and physical-distance evaluation.
                original_probs = np.stack([
                    cv2.resize(probs[i, c].astype(np.float32), (original_w, original_h),
                               interpolation=cv2.INTER_LINEAR)
                    for c in range(2)
                ], axis=0)
                pred_acl = original_probs[0] >= args.threshold
                pred_pcl = original_probs[1] >= args.threshold
                gt_acl = original_mask == 1
                gt_pcl = original_mask == 2
                acl_overlap = segmentation_metrics(pred_acl.astype(np.uint8), gt_acl.astype(np.uint8))
                pcl_overlap = segmentation_metrics(pred_pcl.astype(np.uint8), gt_pcl.astype(np.uint8))
                spacing_y = float(batch["spacing_y_mm"][i])
                spacing_x = float(batch["spacing_x_mm"][i])
                if not np.isfinite(spacing_y) or not np.isfinite(spacing_x):
                    raise ValueError("Physical ASD/HD95 require spacing_y_mm and spacing_x_mm.")
                acl_surface = surface_distances_mm(pred_acl, gt_acl, spacing_y, spacing_x)
                pcl_surface = surface_distances_mm(pred_pcl, gt_pcl, spacing_y, spacing_x)
                rows.append({
                    "subject_id": batch["subject_id"][i],
                    "slice_id": batch["slice_id"][i],
                    "dsc_acl": acl_overlap["dsc"], "iou_acl": acl_overlap["iou"],
                    "dsc_pcl": pcl_overlap["dsc"], "iou_pcl": pcl_overlap["iou"],
                    "asd_acl_mm": acl_surface["asd_mm"], "hd95_acl_mm": acl_surface["hd95_mm"],
                    "asd_pcl_mm": pcl_surface["asd_mm"], "hd95_pcl_mm": pcl_surface["hd95_mm"],
                })
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No test predictions were produced.")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} slice-level rows to {out_path}")
    print("For paper-level subject statistics, aggregate per-slice results by subject first.")


if __name__ == "__main__":
    main()
