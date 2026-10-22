"""Aggregate slice-level metric CSVs to subject-level summaries and CIs.

Input is produced by scripts/evaluate.py or the equivalent baseline evaluator.
The bootstrap unit is the subject, not the slice. This script computes per-
subject mean over the 10 selected slices, then percentile bootstrap CIs.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from evaluation.statistical_tests import bootstrap_mean_ci
METRICS=['dsc_acl','iou_acl','asd_acl_mm','hd95_acl_mm','dsc_pcl','iou_pcl','asd_pcl_mm','hd95_pcl_mm']
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input-csv',required=True); p.add_argument('--output-subject-csv',required=True); p.add_argument('--output-summary-csv',required=True); p.add_argument('--bootstrap',type=int,default=10000); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
 with open(a.input_csv,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
 if not rows: raise ValueError('Input CSV is empty')
 by={}
 for r in rows:
  sid=r['subject_id']; by.setdefault(sid,{m:[] for m in METRICS})
  for m in METRICS:
   v=float(r[m]);
   if np.isfinite(v): by[sid][m].append(v)
 subject=[]
 for sid,vals in sorted(by.items()):
  rr={'subject_id':sid}
  for m in METRICS: rr[m]=float(np.mean(vals[m])) if vals[m] else float('nan')
  subject.append(rr)
 Path(a.output_subject_csv).parent.mkdir(parents=True,exist_ok=True)
 with open(a.output_subject_csv,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['subject_id']+METRICS); w.writeheader(); w.writerows(subject)
 summary=[]
 for i,m in enumerate(METRICS):
  x=[r[m] for r in subject if np.isfinite(r[m])]
  if not x: continue
  mean,lo,hi=bootstrap_mean_ci(x,a.bootstrap,0.95,a.seed+i)
  summary.append({'metric':m,'n_subjects':len(x),'mean':mean,'ci95_low':lo,'ci95_high':hi})
 with open(a.output_summary_csv,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['metric','n_subjects','mean','ci95_low','ci95_high']); w.writeheader(); w.writerows(summary)
 print(f'Aggregated {len(rows)} slices from {len(subject)} subjects; wrote subject-level metrics and bootstrap CIs.')
if __name__=='__main__': main()
