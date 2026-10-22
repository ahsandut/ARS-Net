"""Create a fixed subject-level 160/40 split and five CV manifests.

This script randomly holds out 20% of subjects as independent test subjects, then makes five folds from the remaining subjects.
Keep the generated manifests fixed and use them for every method/ablation.
"""
from __future__ import annotations
import argparse,csv,random
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--test-fraction',type=float,default=.2);p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 if a.folds!=5: print('Warning: manuscript protocol specifies five-fold CV.')
 with open(a.input_csv,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f)); fields=list(rows[0].keys()) if rows else []
 required={'subject_id','image_path','mask_path'}
 if not rows or not required.issubset(fields): raise ValueError(f'Input CSV must contain {sorted(required)}')
 subject_rows={}
 for r in rows: subject_rows.setdefault(r['subject_id'],[]).append(r)
 subjects=sorted(subject_rows); rng=random.Random(a.seed); rng.shuffle(subjects)
 n_test=max(1,round(len(subjects)*a.test_fraction)); test=set(subjects[:n_test]); dev=subjects[n_test:]
 if len(dev)<a.folds: raise ValueError('Not enough development subjects for requested folds')
 # Deterministic round-robin fold assignment after seeded shuffle.
 fold_of={s:i%a.folds for i,s in enumerate(dev)}; out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 for fold in range(a.folds):
  fold_rows=[]
  for sid in subjects:
   split='test' if sid in test else ('val' if fold_of[sid]==fold else 'train')
   for r in subject_rows[sid]: fold_rows.append({**r,'split':split})
  path=out/f'fold{fold+1}.csv'
  with path.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=fields+(['split'] if 'split' not in fields else []));w.writeheader();w.writerows(fold_rows)
  print(f'{path}: train={sum(fold_of[s]!=fold for s in dev)} development subjects, val={sum(fold_of[s]==fold for s in dev)}, test={len(test)}')
 print(f'Fixed split: {len(dev)} development subjects, {len(test)} independent test subjects. Review and preserve these manifests for all methods.')
if __name__=='__main__':main()
