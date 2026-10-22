"""Evaluate an external baseline checkpoint on one manifest split."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from datasets.dataset import KneeLigamentDataset
from datasets.preprocessing import read_mask
from baselines.adapter import load_factory,segmentation_logits
from evaluation.metrics import segmentation_metrics
from evaluation.physical_distances import surface_distances_mm

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--image-root',default='.');p.add_argument('--checkpoint',required=True);p.add_argument('--model-factory',required=True);p.add_argument('--model-kwargs',default='{}');p.add_argument('--split',default='test');p.add_argument('--output-csv',required=True);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--threshold',type=float,default=.5);a=p.parse_args()
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ckpt=torch.load(a.checkpoint,map_location=device); image_size=int(ckpt.get('args',{}).get('image_size',320)) if isinstance(ckpt,dict) else 320
 ds=KneeLigamentDataset(a.manifest,a.split,a.image_root,image_size=image_size,augment=False); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False)
 model=load_factory(a.model_factory,json.loads(a.model_kwargs)).to(device); model.load_state_dict(ckpt.get('model_state',ckpt));model.eval(); root=Path(a.image_root).resolve();rows=[]
 with torch.no_grad():
  for batch in loader:
   logits=segmentation_logits(model(batch['image'].to(device)))
   probs=torch.sigmoid(torch.nn.functional.interpolate(logits,size=batch['image'].shape[-2:],mode='bilinear',align_corners=False)).cpu().numpy()
   for i in range(len(probs)):
    mp=Path(batch['mask_path'][i]);mp=mp if mp.is_absolute() else root/mp;gtcat=read_mask(mp);h,w=gtcat.shape
    restored=np.stack([cv2.resize(probs[i,c].astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR) for c in range(2)])
    pa=restored[0]>=a.threshold;pp=restored[1]>=a.threshold;ga=gtcat==1;gp=gtcat==2
    sy=float(batch['spacing_y_mm'][i]);sx=float(batch['spacing_x_mm'][i])
    if not np.isfinite(sy) or not np.isfinite(sx): raise ValueError('Physical metrics require original DICOM spacing_y_mm and spacing_x_mm.')
    ma=segmentation_metrics(pa.astype(np.uint8),ga.astype(np.uint8));mpcl=segmentation_metrics(pp.astype(np.uint8),gp.astype(np.uint8));da=surface_distances_mm(pa,ga,sy,sx);dp=surface_distances_mm(pp,gp,sy,sx)
    rows.append({'subject_id':batch['subject_id'][i],'slice_id':batch['slice_id'][i],'dsc_acl':ma['dsc'],'iou_acl':ma['iou'],'dsc_pcl':mpcl['dsc'],'iou_pcl':mpcl['iou'],'asd_acl_mm':da['asd_mm'],'hd95_acl_mm':da['hd95_mm'],'asd_pcl_mm':dp['asd_mm'],'hd95_pcl_mm':dp['hd95_mm']})
 if not rows: raise RuntimeError('No samples were evaluated; check the manifest split.')
 out=Path(a.output_csv);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(f'Wrote {len(rows)} slice-level rows to {out}; aggregate by subject before CIs or paired tests.')
if __name__=='__main__':main()
