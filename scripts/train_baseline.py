"""Train an externally implemented SOTA segmentation baseline on one fold."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from datasets.dataset import KneeLigamentDataset, load_manifest, validate_subject_splits
from baselines.adapter import load_factory, segmentation_logits


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def seg_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    den = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - ((2 * inter + 1e-6) / (den + 1e-6)).mean()
    return bce + dice


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True); p.add_argument('--image-root',default='.')
    p.add_argument('--model-factory',required=True,help='python.module:factory')
    p.add_argument('--model-kwargs',default='{}',help='JSON object passed to factory')
    p.add_argument('--method',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--image-size',type=int,default=320); p.add_argument('--epochs',type=int,default=150); p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--warmup-epochs',type=int,default=5); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--amp',action='store_true')
    args=p.parse_args(); seed_everything(args.seed)
    rows=load_manifest(args.manifest); validate_subject_splits(rows)
    train_ds=KneeLigamentDataset(args.manifest,'train',args.image_root,image_size=args.image_size,augment=True,seed=args.seed)
    val_ds=KneeLigamentDataset(args.manifest,'val',args.image_root,image_size=args.image_size,augment=False,seed=args.seed)
    if {r['subject_id'] for r in train_ds.rows} & {r['subject_id'] for r in val_ds.rows}:
        raise RuntimeError('Subject leakage between train and validation')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=device.type=='cuda')
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
    model=load_factory(args.model_factory,json.loads(args.model_kwargs)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler=torch.cuda.amp.GradScaler(enabled=args.amp and device.type=='cuda')
    def lr_factor(epoch):
        if epoch < args.warmup_epochs: return float(epoch+1)/max(1,args.warmup_epochs)
        progress=(epoch-args.warmup_epochs)/max(1,args.epochs-args.warmup_epochs)
        return 0.5*(1+np.cos(np.pi*min(progress,1.0)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'run_config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    best=float('inf')
    for epoch in range(args.epochs):
        model.train(); tr=[]
        for batch in train_loader:
            x=batch['image'].to(device); y=batch['mask'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type=='cuda'):
                logits=segmentation_logits(model(x))
                if logits.shape[-2:] != y.shape[-2:]: logits=F.interpolate(logits,size=y.shape[-2:],mode='bilinear',align_corners=False)
                loss=seg_loss(logits,y)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
            tr.append(float(loss.detach().cpu()))
        model.eval(); va=[]
        with torch.no_grad():
            for batch in val_loader:
                x=batch['image'].to(device); y=batch['mask'].to(device)
                logits=segmentation_logits(model(x))
                if logits.shape[-2:] != y.shape[-2:]: logits=F.interpolate(logits,size=y.shape[-2:],mode='bilinear',align_corners=False)
                va.append(float(seg_loss(logits,y).cpu()))
        scheduler.step(); val=float(np.mean(va))
        state={'epoch':epoch+1,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
               'best_val_loss':min(best,val),'method':args.method,'model_factory':args.model_factory,
               'model_kwargs':json.loads(args.model_kwargs),'args':vars(args)}
        torch.save(state,out/'last.pt')
        if val<best: best=val; state['best_val_loss']=best; torch.save(state,out/'best.pt')
        print(f"{args.method} epoch {epoch+1}/{args.epochs}: train={np.mean(tr):.5f} val={val:.5f}",flush=True)

if __name__=='__main__': main()
