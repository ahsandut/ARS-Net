"""Orchestrate named SOTA baselines across subject-level CV manifests.

Registry JSON format:
{"TransUNet":{"factory":"external_transunet.factory:build_model","kwargs":{"num_classes":2}}, ...}
Each fold manifest must have split labels train/val/test and subject-disjoint
train/val records. Use the same fold manifests for ARS-Net and every baseline.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_METHODS=[]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--registry',required=True); p.add_argument('--manifest-template',required=True,help='e.g. manifests/fold{fold}.csv')
    p.add_argument('--image-root',default='data/private_dataset'); p.add_argument('--output-root',default='outputs/sota_cv'); p.add_argument('--methods',default='',help='Comma-separated methods enabled in the registry; defaults to all enabled entries')
    p.add_argument('--image-size',type=int,default=320); p.add_argument('--epochs',type=int,default=150); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--warmup-epochs',type=int,default=5); p.add_argument('--seed',type=int,default=42); p.add_argument('--amp',action='store_true')
    a=p.parse_args(); registry_path=Path(a.registry); registry_path=registry_path if registry_path.is_absolute() else ROOT/registry_path; registry=json.loads(registry_path.read_text(encoding='utf-8')); methods=[m.strip() for m in a.methods.split(',') if m.strip()] or [name for name,cfg in registry.items() if cfg.get('enabled', False)]
    if not methods: raise ValueError('No enabled methods in registry. See configs/baselines/registry.json and docs/sota_comparison.md.')
    for method in methods:
        if method not in registry: raise KeyError(f'{method!r} missing from registry {a.registry}')
        cfg=registry[method]
        if not cfg.get('enabled', False): raise ValueError(f'{method}: this method is not enabled in the local registry; see configs/baselines/external_sources.json')
        if not cfg.get('factory'): raise ValueError(f'{method}: registry entry has no runnable factory')
        for fold in range(1,6):
            manifest=a.manifest_template.format(fold=fold); manifest_path=Path(manifest); manifest=str(manifest_path if manifest_path.is_absolute() else ROOT/manifest_path)
            image_root=Path(a.image_root); image_root=str(image_root if image_root.is_absolute() else ROOT/image_root)
            out=Path(a.output_root)/method.replace('/','_')/f'fold{fold}'
            cmd=[sys.executable,str(ROOT/'scripts'/'train_baseline.py'),'--manifest',manifest,'--image-root',image_root,
                 '--model-factory',cfg['factory'],'--model-kwargs',json.dumps(cfg.get('kwargs',{})),'--method',method,'--output-dir',str(out),
                 '--image-size',str(a.image_size),'--epochs',str(a.epochs),'--batch-size',str(a.batch_size),'--workers',str(a.workers),'--lr',str(a.lr),
                 '--weight-decay',str(a.weight_decay),'--warmup-epochs',str(a.warmup_epochs),'--seed',str(a.seed)]
            if a.amp: cmd.append('--amp')
            print('\nLaunching',method,'fold',fold,flush=True); subprocess.run(cmd,check=True,cwd=ROOT)
if __name__=='__main__': main()
