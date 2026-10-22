# Private dataset placement (do not commit data)

Place authorized, de-identified 2D image/mask exports under this exact structure:

```text
data/private_dataset/
├── images/
│   ├── SUBJ001/
│   │   ├── 000.png
│   │   └── 001.png
│   └── SUBJ002/
│       └── 000.png
├── masks/
│   ├── SUBJ001/
│   │   ├── 000.png
│   │   └── 001.png
│   └── SUBJ002/
│       └── 000.png
└── metadata.csv
```

Image and mask files must have matching relative paths. Images are grayscale. Masks are categorical uint8 PNGs: `0=background`, `1=ACL`, `2=PCL`. 

`metadata.csv` is required and must have these exact columns:

```csv
subject_id,slice_id,spacing_y_mm,spacing_x_mm
SUBJ001,000,0.3125,0.3125
SUBJ001,001,0.3125,0.3125
SUBJ002,000,0.3125,0.3125
```
