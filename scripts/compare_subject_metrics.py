"""Paired subject-level comparisons between proposed model and comparators."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from evaluation.statistical_tests import paired_wilcoxon,holm_adjust
METRICS=['dsc_acl','iou_acl','asd_acl_mm','hd95_acl_mm','dsc_pcl','iou_pcl','asd_pcl_mm','hd95_pcl_mm']
def load(path):
 with open(path,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
 return {r['subject_id']:r for r in rows}
def main():
 p=argparse.ArgumentParser();p.add_argument('--proposed',required=True);p.add_argument('--comparators',nargs='+',required=True,help='METHOD=subject_metrics.csv');p.add_argument('--output-csv',required=True);a=p.parse_args()
 prop=load(a.proposed); results=[]
 for spec in a.comparators:
  name,path=spec.split('=',1); comp=load(path); ids=sorted(set(prop)&set(comp))
  if len(ids)<2: raise ValueError(f'{name}: need at least two matched subjects')
  for metric in METRICS:
   aa=np.array([float(prop[s][metric]) for s in ids]);bb=np.array([float(comp[s][metric]) for s in ids]); keep=np.isfinite(aa)&np.isfinite(bb)
   if keep.sum()<2: continue
   test=paired_wilcoxon(aa[keep],bb[keep]); results.append({'method':name,'metric':metric,'n_subjects':test['n'],'median_delta_proposed_minus_comparator':test['median_delta'],'p_raw':test['p_raw']})
 # Holm correction across all method-metric comparisons, as a conservative family.
 adjusted=holm_adjust([r['p_raw'] for r in results]) if results else []
 for r,padj in zip(results,adjusted): r['p_holm']=float(padj)
 Path(a.output_csv).parent.mkdir(parents=True,exist_ok=True)
 with open(a.output_csv,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['method','metric','n_subjects','median_delta_proposed_minus_comparator','p_raw','p_holm']);w.writeheader();w.writerows(results)
 print(f'Wrote {len(results)} paired comparisons to {a.output_csv}')
if __name__=='__main__':main()
