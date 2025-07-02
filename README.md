# ReHyDIL ：Hypergraph Tversky-Aware Domain Incremental Learning for Brain Tumor Segmentation with Missing Modalities

## 📂 Dataset Preparation

### 1. Download the Dataset

This demo requires the **BraTS 2019 (BraTS19)** dataset. You can request access and download it from the official source:

* [**The Brain Tumor Segmentation (BraTS) Challenge**](https://www.med.upenn.edu/cbica/brats2019/registration.html)

We provide a simple demo for testing the code.

In the demo, incremental learning can be done using the following commands:  
python demo.py --img_mode t1  
python demo.py --img_mode t2 --prev_img_mode t1  
python demo.py --img_mode flair --prev_img_mode t2-t1  
python demo.py --img_mode t1ce --prev_img_mode flair-t2-t1  

The BraTS19 dataset needs to be used, and the preprocessing method is as described in the paper.

The complete code and documentation will be released after acceptance.
