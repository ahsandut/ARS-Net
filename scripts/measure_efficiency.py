"""Measure ARS-Net parameters and inference latency."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from models import ARSNet
from evaluation.efficiency import count_parameters,measure_latency_ms,estimate_flops_macs

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--architecture-variant',default='full',choices=['full','no_arrm','no_msfm','no_arrm_no_msfm']);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--iterations',type=int,default=100);p.add_argument('--warmup',type=int,default=20);a=p.parse_args()
 device=torch.device(a.device);model=ARSNet(architecture_variant=a.architecture_variant).to(device).eval()
 if a.checkpoint:
  ckpt=torch.load(a.checkpoint,map_location=device);variant=ckpt.get('args',{}).get('architecture_variant',a.architecture_variant);model=ARSNet(architecture_variant=variant).to(device).eval();model.load_state_dict(ckpt.get('model_state',ckpt))
 x=torch.randn(1,1,320,320,device=device);params=count_parameters(model)
 latency=measure_latency_ms(model,x,warmup_iterations=a.warmup,measured_iterations=a.iterations)
 print(f"Parameters: {params['parameters']:,} ({params['parameters']/1e6:.3f} M); trainable={params['trainable_parameters']:,}");print(f"Latency: mean={latency['mean_latency_ms']:.3f} ms, median={latency['median_latency_ms']:.3f} ms on {device}")
 try: print(f"MACs/FLOPs estimate: {estimate_flops_macs(model,x)} (tool-dependent; report convention explicitly)")
 except Exception as e: print(f'FLOPs estimate unavailable: {e}')
if __name__=='__main__':main()
