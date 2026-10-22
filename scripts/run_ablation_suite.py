"""Launch ARS-Net architecture and loss ablations as separate training runs.

The script intentionally launches one process per run, stores each run in its
own output folder, and stops on the first failed run. It reuses the manifest's
existing train/val subject split; for five-fold CV, prepare a separate manifest
for each fold and run the suite per fold.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_VARIANTS = ["full", "no_arrm", "no_msfm", "no_arrm_no_msfm"]
LOSS_VARIANTS = ["L1", "L2", "L3", "L4"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--output-root", default="outputs/ablations")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--suite", choices=["architecture", "loss", "all"], default="all")
    args = parser.parse_args()

    runs: list[tuple[str, str]] = []
    if args.suite in ("architecture", "all"):
        runs.extend((v, "L4") for v in ARCHITECTURE_VARIANTS)
    if args.suite in ("loss", "all"):
        runs.extend(("full", v) for v in LOSS_VARIANTS)
    # Full + L4 appears in both groups; run it only once.
    runs = list(dict.fromkeys(runs))

    for arch, loss in runs:
        run_name = f"arch-{arch}__loss-{loss}"
        out_dir = Path(args.output_root) / run_name
        cmd = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--manifest", args.manifest,
            "--image-root", args.image_root,
            "--output-dir", str(out_dir),
            "--image-size", str(args.image_size),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--warmup-epochs", str(args.warmup_epochs),
            "--seed", str(args.seed),
            "--architecture-variant", arch,
            "--loss-variant", loss,
        ]
        if args.amp:
            cmd.append("--amp")
        print("\nLaunching:", run_name, flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"Completed {len(runs)} ablation runs. Outputs: {Path(args.output_root).resolve()}")

if __name__ == "__main__":
    main()
