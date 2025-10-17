# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk
import skimage.io as io
import warnings
warnings.filterwarnings("ignore")

flair_name = "_flair.nii.gz"
t1_name    = "_t1.nii.gz"
t1ce_name  = "_t1ce.nii.gz"
t2_name    = "_t2.nii.gz"
mask_name  = "_seg.nii.gz"

root_dir = '/path/to/data_root'
out_root = "/path/to/data_root/BraTS_slice"

outputFlair_path   = os.path.join(out_root, "imgs_flair")
outputT1_path      = os.path.join(out_root, "imgs_t1")
outputT2_path      = os.path.join(out_root, "imgs_t2")
outputT1ce_path    = os.path.join(out_root, "imgs_t1ce")
outputMaskWT_path  = os.path.join(out_root, "masks")
outputMaskAll_path = os.path.join(out_root, "masks_all")


SKIP_EMPTY_SLICES = False

CROP_SIZE = 224


def make_output_dirs():
    for p in [outputFlair_path, outputT1_path, outputT2_path, outputT1ce_path,
              outputMaskWT_path, outputMaskAll_path]:
        os.makedirs(p, exist_ok=True)


def list_case_ids(brats_root):
    names = [d for d in os.listdir(brats_root)
             if os.path.isdir(os.path.join(brats_root, d)) and not d.startswith('.')]
    names.sort()
    return names


def modality_paths(case_root, case_id):
    def p(suffix): return os.path.join(case_root, case_id, case_id + suffix)
    paths = {
        "flair": p(flair_name),
        "t1":    p(t1_name),
        "t1ce":  p(t1ce_name),
        "t2":    p(t2_name),
        "seg":   p(mask_name),
    }
    for k, v in paths.items():
        if not os.path.exists(v):
            raise FileNotFoundError(f"[{case_id}] missing modality: {k} -> {v}")
    return paths


def read_nii(path, sitk_type=None):
    if sitk_type is None:
        img = sitk.ReadImage(path)
    else:
        img = sitk.ReadImage(path, sitk_type)
    arr = sitk.GetArrayFromImage(img)  # (Z, H, W)
    return arr


def robust_norm(volume):
    v = volume.astype(np.float32)
    nz = v[v > 0]
    if nz.size > 0:
        lo, hi = np.percentile(nz, [1, 99])
        v = np.clip(v, lo, hi)
        nz = v[v > 0]
        if nz.size > 0 and nz.std() > 0:
            v = (v - nz.mean()) / (nz.std() + 1e-8)
    v_min, v_max = v.min(), v.max()
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min + 1e-8)
    else:
        v = np.zeros_like(v, dtype=np.float32)
    return v


def crop_to_224_centered_on_mask(volumes, mask, size=224):
    assert len(volumes) > 0
    Z, H, W = mask.shape
    union2d = (mask > 0).any(axis=0).astype(np.uint8)  # (H,W)

    if union2d.any():
        ys, xs = np.where(union2d > 0)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    else:
        cy, cx = H // 2, W // 2

    half = size // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    pad_y0, pad_y1 = max(0, -y0), max(0, y1 - H)
    pad_x0, pad_x1 = max(0, -x0), max(0, x1 - W)

    def pad_and_crop(arr):
        if pad_y0 or pad_y1 or pad_x0 or pad_x1:
            arr = np.pad(arr,
                         ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)),
                         mode='constant', constant_values=0)
        yy0, yy1 = y0 + pad_y0, y1 + pad_y0
        xx0, xx1 = x0 + pad_x0, x1 + pad_x0
        return arr[:, yy0:yy1, xx0:xx1]

    out_vols = [pad_and_crop(v) for v in volumes]
    out_mask = pad_and_crop(mask)
    for v in out_vols + [out_mask]:
        assert v.shape[1] == size and v.shape[2] == size, f"Crop failed: got {v.shape}"
    return out_vols, out_mask


def build_labels_3ch(mask_slice):
    WT = (mask_slice > 0).astype(np.uint8)
    TC = ((mask_slice == 1) | (mask_slice == 4)).astype(np.uint8)
    ET = (mask_slice == 4).astype(np.uint8)
    all_label3 = np.stack([WT, TC, ET], axis=0).astype(np.uint8)
    return WT, all_label3


def to_slice_id(n):
    s = str(n)
    if len(s) == 1: return '00' + s
    if len(s) == 2: return '0' + s
    return s


def main():
    make_output_dirs()

    case_ids = list_case_ids(root_dir)
    print("train_hgg_list:", len(case_ids), case_ids, '\n')
    all_list = case_ids
    print("\n all_list:", len(all_list), all_list)

    for idx, case_id in enumerate(all_list):
        try:
            paths = modality_paths(root_dir, case_id)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        print(f"[{idx+1}/{len(all_list)}] Processing case: {case_id}")

        flair = read_nii(paths["flair"], sitk.sitkInt16)
        t1    = read_nii(paths["t1"],    sitk.sitkInt16)
        t1ce  = read_nii(paths["t1ce"],  sitk.sitkInt16)
        t2    = read_nii(paths["t2"],    sitk.sitkInt16)
        seg   = read_nii(paths["seg"],   sitk.sitkUInt8)


        shapes = {tuple(flair.shape), tuple(t1.shape), tuple(t1ce.shape), tuple(t2.shape), tuple(seg.shape)}
        if len(shapes) != 1:
            print(f"[WARN] Shape mismatch in {case_id}: {shapes}. Skipped.")
            continue

        flair_n = robust_norm(flair)
        t1_n    = robust_norm(t1)
        t1ce_n  = robust_norm(t1ce)
        t2_n    = robust_norm(t2)

        (flair_c, t1_c, t1ce_c, t2_c), seg_c = crop_to_224_centered_on_mask(
            [flair_n, t1_n, t1ce_n, t2_n], seg, size=CROP_SIZE
        )

        Z = seg_c.shape[0]
        for z in range(Z):
            mask_np = seg_c[z, :, :]

            if SKIP_EMPTY_SLICES and (mask_np.max() == 0):
                continue

            flair_np = flair_c[z, :, :].astype(np.float32)
            t1_np    = t1_c[z, :, :].astype(np.float32)
            t1ce_np  = t1ce_c[z, :, :].astype(np.float32)
            t2_np    = t2_c[z, :, :].astype(np.float32)

            WT_Label, all_label3 = build_labels_3ch(mask_np)
            mask_np_wt = WT_Label.astype(np.uint8)

            slice_id = to_slice_id(z + 1)

            flair_imagepath = os.path.join(outputFlair_path,  f"{case_id}_{slice_id}.npy")
            t1_imagepath    = os.path.join(outputT1_path,     f"{case_id}_{slice_id}.npy")
            t2_imagepath    = os.path.join(outputT2_path,     f"{case_id}_{slice_id}.npy")
            t1ce_imagepath  = os.path.join(outputT1ce_path,   f"{case_id}_{slice_id}.npy")
            maskpath_wt     = os.path.join(outputMaskWT_path, f"{case_id}_{slice_id}.npy")
            maskpath_all    = os.path.join(outputMaskAll_path,f"{case_id}_{slice_id}.npy")


            np.save(flair_imagepath, flair_np)       # float32, (224,224)
            np.save(t1_imagepath,    t1_np)          # float32, (224,224)
            np.save(t2_imagepath,    t2_np)          # float32, (224,224)
            np.save(t1ce_imagepath,  t1ce_np)        # float32, (224,224)
            np.save(maskpath_wt,     mask_np_wt)     # uint8,   (224,224),
            np.save(maskpath_all,    all_label3)     # uint8,   (3,224,224), WT/TC/ET

        print(f"  -> Saved slices for {case_id}: {Z} (kept all = {not SKIP_EMPTY_SLICES})")

    print("Done！")


if __name__ == "__main__":
    main()
