"""Run trained ARS-Net inference and save categorical masks as PNG."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from datasets.preprocessing import read_image, resize_pair, zscore_normalize
from models import ARSNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    image_size = int(ckpt.get("args", {}).get("image_size", 320)) if isinstance(ckpt, dict) else 320
    original = read_image(args.image)
    resized, _ = resize_pair(original, np.zeros_like(original, dtype=np.uint8), (image_size, image_size))
    normalized = zscore_normalize(resized)
    x = torch.from_numpy(normalized[None, None]).float().to(device)
    architecture_variant = ckpt.get("args", {}).get("architecture_variant", "full") if isinstance(ckpt, dict) else "full"
    model = ARSNet(input_size=image_size, architecture_variant=architecture_variant).to(device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x)["segmentation_logits"])[0].cpu().numpy()
    acl = probs[0] >= args.threshold
    pcl = probs[1] >= args.threshold
    categorical = np.zeros(probs.shape[1:], dtype=np.uint8)
    only_acl = acl & ~pcl
    only_pcl = pcl & ~acl
    overlap = acl & pcl
    categorical[only_acl] = 1
    categorical[only_pcl] = 2
    categorical[overlap & (probs[0] >= probs[1])] = 1
    categorical[overlap & (probs[1] > probs[0])] = 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), categorical)
    print(f"Saved categorical mask (0=background, 1=ACL, 2=PCL) to {out}")


if __name__ == "__main__":
    main()
