# ReHyDIL ：Hypergraph Tversky-Aware Domain Incremental Learning for Brain Tumor Segmentation with Missing Modalities

## 📂 Dataset Preparation

This demo requires the **BraTS 2019 (BraTS19)** dataset. You can request access and download it from the official source:

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

## 🚀 Train

The `train.py` script simulates an incremental learning scenario where the model learns from different MRI modalities sequentially.

The process is divided into four steps. Please run them in the following order:

**Step 1: Train on T1 images**
The model learns its first task using only the T1 modality.
```bash
python train.py --img_mode t1
```
**Step 2: Incrementally add T2 images**
The model, already trained on T1, now learns to incorporate T2 images without forgetting the initial knowledge.
```bash
python train.py --img_mode t2 --prev_img_mode t1  
```

**Step 3: Incrementally add FLAIR images**
The model continues to learn, now adding the FLAIR modality.
```bash
python train.py --img_mode flair --prev_img_mode t2-t1   
```

**Step 4: Incrementally add T1ce images**
Finally, the model learns from the T1ce modality, completing the sequence.
```bash
python train.py --img_mode t1ce --prev_img_mode flair-t2-t1  
```

