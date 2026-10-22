"""Build the dataset manifest."""
from __future__ import annotations
import argparse, csv
from pathlib import Path

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy'}
REQUIRED_META = {'subject_id', 'slice_id', 'spacing_y_mm', 'spacing_x_mm'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', default='data/private_dataset')
    p.add_argument('--output', default='manifests/all_slices.csv')
    p.add_argument('--metadata', default=None, help='Defaults to <dataset-root>/metadata.csv')
    a = p.parse_args()
    root = Path(a.dataset_root).resolve()
    images_root, masks_root = root/'images', root/'masks'
    meta_path = Path(a.metadata).resolve() if a.metadata else root/'metadata.csv'
    for required in (images_root, masks_root, meta_path):
        if not required.exists():
            raise FileNotFoundError(f'Required dataset path does not exist: {required}')
    with meta_path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED_META.issubset(reader.fieldnames):
            raise ValueError(f'{meta_path} must contain columns {sorted(REQUIRED_META)}')
        metadata = {}
        for row in reader:
            key = (row['subject_id'].strip(), row['slice_id'].strip())
            if key in metadata:
                raise ValueError(f'Duplicate metadata row for subject/slice {key}')
            try:
                sy, sx = float(row['spacing_y_mm']), float(row['spacing_x_mm'])
            except (TypeError, ValueError):
                raise ValueError(f'Invalid pixel spacing for subject/slice {key}')
            if sy <= 0 or sx <= 0:
                raise ValueError(f'Pixel spacing must be positive for subject/slice {key}')
            metadata[key] = (sy, sx)
    image_paths = sorted(q for q in images_root.rglob('*') if q.is_file() and q.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise RuntimeError(f'No supported image files found under {images_root}')
    rows, seen = [], set()
    for image_path in image_paths:
        rel = image_path.relative_to(images_root)
        if len(rel.parts) < 2:
            raise ValueError(f'Images must be grouped by subject folder: {image_path}')
        subject_id, slice_name = rel.parts[0], rel.name
        slice_id = image_path.stem
        key = (subject_id, slice_id)
        if key in seen:
            raise ValueError(f'Duplicate image key {key}; use unique slice filenames per subject')
        seen.add(key)
        mask_path = masks_root / rel
        if not mask_path.exists():
            raise FileNotFoundError(f'Missing matching mask for image {image_path}: expected {mask_path}')
        if key not in metadata:
            raise KeyError(f'Missing metadata.csv row for subject_id,slice_id={key}')
        sy, sx = metadata[key]
        rows.append({'subject_id': subject_id, 'slice_id': slice_id,
                     'image_path': (Path('images') / rel).as_posix(), 'mask_path': (Path('masks') / rel).as_posix(),
                     'spacing_y_mm': f'{sy:.8g}', 'spacing_x_mm': f'{sx:.8g}', 'split': 'all'})
    extras = set(metadata) - seen
    if extras:
        raise ValueError(f'Metadata contains {len(extras)} subject/slice entries without images; first entries: {sorted(extras)[:5]}')
    subjects = {r['subject_id'] for r in rows}
    out = Path(a.output).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'Wrote {len(rows)} slices from {len(subjects)} subjects to {out}')
    print('Verify the subject count, label encoding, spacing, and all paths before creating fixed CV manifests.')

if __name__ == '__main__':
    main()
