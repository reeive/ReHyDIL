# Hypergraph Tversky-Aware Domain Incremental Learning for Brain Tumor Segmentation with Missing Modalities

* [MICCAI 2025](https://link.springer.com/chapter/10.1007/978-3-032-05141-7_28) ReHyDIL — **Re**play + **Hy**pergraph **D**omain **I**ncremental **L**earning
  
* [Paper link](https://papers.miccai.org/miccai-2025/paper/2774_paper.pdf)

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

`train.py` is a stage-wise runner: it trains the model incrementally over MRI modalities.
Default configuration follows the clinical order: `t1, t2, flair, t1ce`.

If you don’t pass `--stages`, the script will run all four stages in that order.

**Quick start (full clinical sequence — default)**
The model learns its first task using only the T1 modality.
```bash
python train.py \
  --data_path /path/to/data_root \
  --out_root  /path/to/outputs \
  --train_fmt /path/lists/train.list \
  --val_fmt   /path/lists/val.list
```

