"""Validate enabled local baseline factories and their two-class output contract."""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from baselines.adapter import load_factory, segmentation_logits


def main():
    path = ROOT/'configs/baselines/registry.json'
    registry = json.loads(path.read_text(encoding='utf-8'))
    enabled = [(name, cfg) for name, cfg in registry.items() if cfg.get('enabled')]
    if not enabled:
        raise RuntimeError(f'No enabled baseline factories in {path}')
    for name, cfg in enabled:
        if not cfg.get('factory'):
            raise ValueError(f'{name}: enabled entry must define a factory')
        model = load_factory(cfg['factory'], cfg.get('kwargs', {})).cpu().eval()
        with torch.no_grad():
            logits = segmentation_logits(model(torch.zeros(1, 1, 64, 64)))
        if logits.shape != (1, 2, 64, 64):
            raise ValueError(f'{name}: expected (1,2,64,64), got {tuple(logits.shape)}')
        print(f'[PASS] {name}: factory={cfg["factory"]}, output={tuple(logits.shape)}')
    print(f'Validated {len(enabled)} enabled baseline(s). This is a shape test, not an accuracy/training result.')

if __name__ == '__main__':
    main()
