"""Clone the official baseline repositories into fixed, repository-relative paths."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', default='', help='Comma-separated keys from configs/baselines/external_sources.json')
    p.add_argument('--update-existing', action='store_true', help='Run git pull for existing repositories')
    a = p.parse_args()
    sources = json.loads((ROOT/'configs/baselines/external_sources.json').read_text(encoding='utf-8'))
    selected = [x.strip() for x in a.only.split(',') if x.strip()] or list(sources)
    unknown = sorted(set(selected)-set(sources))
    if unknown:
        raise KeyError(f'Unknown baseline names {unknown}; available: {list(sources)}')
    for name in selected:
        cfg = sources[name]
        target = ROOT / cfg['local_path']
        if (target/'.git').exists():
            if a.update_existing:
                subprocess.run(['git','-C',str(target),'pull','--ff-only'],check=True)
            else:
                print(f'[EXISTS] {name}: {target} (not updated; pass --update-existing to pull)')
            continue
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f'{target} exists but is not a Git checkout.')
        target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--depth','1',cfg['repository'],str(target)],check=True)
        print(f'[CLONED] {name}: {target}')
    print('\nSource download complete.')

if __name__ == '__main__':
    main()
