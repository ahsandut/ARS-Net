"""Create MRI/GT/prediction/overlay/error-map panels from a trained ARS-Net."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import cv2,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from datasets.preprocessing import read_image,read_mask,resize_pair,zscore_normalize
from models import ARSNet

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--mask',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--threshold',type=float,default=.5);a=p.parse_args()
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');image=read_image(a.image);gt=read_mask(a.mask);resized,_=resize_pair(image,np.zeros_like(image,dtype=np.uint8),(320,320));x=torch.from_numpy(zscore_normalize(resized)[None,None].astype(np.float32)).to(device)
 ckpt=torch.load(a.checkpoint,map_location=device);variant=ckpt.get('args',{}).get('architecture_variant','full') if isinstance(ckpt,dict) else 'full';model=ARSNet(architecture_variant=variant).to(device);model.load_state_dict(ckpt.get('model_state',ckpt));model.eval()
 with torch.no_grad(): prob=torch.sigmoid(model(x)['segmentation_logits'])[0].cpu().numpy()
 h,w=gt.shape;prob=np.stack([cv2.resize(prob[c],(w,h),interpolation=cv2.INTER_LINEAR) for c in range(2)]);pred=np.zeros((h,w),np.uint8);pred[prob[0]>=a.threshold]=1;pred[prob[1]>=a.threshold]=2
 # RGB overlays: GT boundaries in yellow, predicted boundaries in cyan/magenta.
 gray=cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR);base=cv2.cvtColor(cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cv2.COLOR_GRAY2RGB)
 overlay=base.copy();gt_a=cv2.Canny((gt==1).astype(np.uint8)*255,0,1)>0;gt_p=cv2.Canny((gt==2).astype(np.uint8)*255,0,1)>0;pr_a=cv2.Canny((pred==1).astype(np.uint8)*255,0,1)>0;pr_p=cv2.Canny((pred==2).astype(np.uint8)*255,0,1)>0
 overlay[gt_a]=(255,255,0);overlay[gt_p]=(255,255,0);overlay[pr_a]=(0,255,255);overlay[pr_p]=(255,0,255)
 # Error map: green true-positive, red false-positive, blue false-negative, combined across ligaments.
 gt_fg=gt>0;pr_fg=pred>0;err=np.zeros((h,w,3),np.uint8);err[gt_fg&pr_fg]=(0,200,0);err[pr_fg&~gt_fg]=(230,40,40);err[gt_fg&~pr_fg]=(40,80,230)
 fig,ax=plt.subplots(1,5,figsize=(16,4));ax[0].imshow(gray,cmap='gray');ax[0].set_title('MRI');ax[1].imshow(gt,cmap='viridis',vmin=0,vmax=2);ax[1].set_title('Ground truth');ax[2].imshow(pred,cmap='viridis',vmin=0,vmax=2);ax[2].set_title('Prediction');ax[3].imshow(overlay);ax[3].set_title('Contours: GT / prediction');ax[4].imshow(err);ax[4].set_title('Error: TP / FP / FN')
 for aa in ax:aa.axis('off')
 fig.tight_layout();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=300,bbox_inches='tight');plt.close(fig);print(f'Saved {out}')
if __name__=='__main__':main()
