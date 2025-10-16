# Hypergraph Tversky-Aware Domain Incremental Learning for Brain Tumor Segmentation with Missing Modalities

## Dataset Preparation

This work requires the **BraTS 2019 (BraTS19)** dataset. You can request access and download it from the official source:

* [**The Brain Tumor Segmentation (BraTS) Challenge**](https://www.med.upenn.edu/cbica/brats2019.html)

The script expects the data to be in a directory named ./BraTS19
```bash
./BraTS19/
├── HGG/
│   ├── BraTS19_TCIA01_.../
│   │   ├── BraTS19_TCIA01_..._flair.nii.gz
│   │   ├── BraTS19_TCIA01_..._t1.nii.gz
│   │   ├── BraTS19_TCIA01_..._t1ce.nii.gz
│   │   ├── BraTS19_TCIA01_..._t2.nii.gz
│   │   └── BraTS19_TCIA01_..._seg.nii.gz
│   └── ...
└── LGG/
    ├── BraTS19_TCIA08_.../
    │   ├── ...
    └── ...
```

## Data & Lists

Create patient-level train/val/test lists from BraTS19. Adjust paths/ratios as needed.

```bash
# Example: write lists to ./lists using a 80/10/10 split (patient-level)
python pre_list.py \
  --data_root ./BraTS19 \
  --out_dir   ./lists \
  --val_ratio 0.10 \
  --test_ratio 0.10 
```

## Train

The `train.py` script simulates an incremental learning scenario where the model learns from different MRI modalities sequentially.

The process is divided into four steps. Please run them in the following order:

**Step 1: Train on T1 images**
The model learns its first task using only the T1 modality.
```bash
python train.py \
  --data_path /path/to/data_root \
  --out_root  /path/to/outputs \
  --stages t1 \
  --train_fmt /path/lists/train.list \
  --val_fmt   /path/lists/val.list
```
**Step 2: Incrementally add T2 images**
The model, already trained on T1, now learns to incorporate T2 images without forgetting the initial knowledge.
```bash
python train.py \
  --data_path /path/to/data_root \
  --out_root  /path/to/outputs \
  --stages t1,t2 \
  --train_fmt /path/lists/train.list \
  --val_fmt   /path/lists/val.list
```

**Step 3: Incrementally add FLAIR images**
The model continues to learn, now adding the FLAIR modality.
```bash
python train.py \
  --data_path /path/to/data_root \
  --out_root  /path/to/outputs \
  --stages t1,t2,flair \
  --train_fmt /path/lists/train.list \
  --val_fmt   /path/lists/val.list  
```

**Step 4: Incrementally add T1ce images**
Finally, the model learns from the T1ce modality, completing the sequence.
```bash
python train.py \
  --data_path /path/to/data_root \
  --out_root  /path/to/outputs \
  --stages t1,t2,flair,t1ce \
  --train_fmt /path/lists/train.list \
  --val_fmt   /path/lists/val.list
```

