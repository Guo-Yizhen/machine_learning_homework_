import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
import os
import pandas as pd
from functools import partial
import argparse
from my_Unet import Unet
from dataset import FracRibTestDateSet
from skimage.measure import label, regionprops
import math

def remove_small_objects(prediction, min_size=512, cnct=3):
    pred_lbl = label(prediction, connectivity=cnct)
    for region_idx in range(1, int(pred_lbl.max() + 1)):
        size = np.where(pred_lbl == region_idx)[0].shape[0]
        if size<=min_size:
            prediction = prediction - (pred_lbl == region_idx).astype(np.int8)
    pred_lbl = label(prediction, connectivity=cnct)
    return prediction, pred_lbl

        
def execute_image(img,img_id,shape,block_dict,output_path, model, device, block_size):
    # idx = idx_data_target[0]
    model.eval()

    l_iter = math.ceil(img.shape[0] / (block_size / 2)) - 1
    w_iter = math.ceil(img.shape[1] / (block_size / 2)) - 1
    d_iter = math.ceil(img.shape[2] / (block_size / 2)) - 1
    with torch.no_grad():
        prediction = np.zeros(shape, dtype=float)
        overlap_num = np.zeros(shape, dtype=float)
        for l in range(l_iter):
            for w in range(w_iter):
                for d in range(d_iter):
                    l_high = int(min((l + 2) * block_size / 2, img.shape[0]))
                    l_low = int(l_high - block_size)
                    w_high = int(min((w + 2) * block_size / 2, img.shape[1]))
                    w_low = int(w_high - block_size)
                    d_high = int(min((d + 2) * block_size / 2, img.shape[2]))
                    d_low = int(d_high - block_size)

                    array = img[l_low:l_high, w_low:w_high, d_low:d_high].astype(np.float)[np.newaxis, :]
                    array = torch.from_numpy(array).float()

                    if array.ndim == 4:
                        array = array[np.newaxis, :]
                    array = array.to(device)
                    output = model(array)
                    output = torch.sigmoid(output).squeeze().cpu().detach().numpy()
                    prediction[l_low:l_high, w_low:w_high, d_low:d_high] = prediction[l_low:l_high, w_low:w_high,
                                                                   d_low:d_high] + output
                    overlap_num[l_low:l_high, w_low:w_high, d_low:d_high] = overlap_num[l_low:l_high, w_low:w_high,
                                                                    d_low:d_high] + 1
        prediction = prediction / overlap_num

        pred_copy = prediction >= 0.5
        # pred_label = label(pred_bin).astype(np.int16)  # 根据pred，做一个简单的聚类（不同簇有不同的标签）
        pred_copy, pred_label = remove_small_objects(pred_copy, cnct=3)
        pred_regions = regionprops(pred_label, prediction)
        pred_index = [0] + [region.label for region in pred_regions]  # 就是0加上各个label
        pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]  # 就是在每个label标出来的连通区域内，计算pred的平均值
        # placeholder for label class since classification isn't included
        pred_label_code = [0] + [1 for i in range(int(pred_label.max()))]  # 随便生成这些区域的label code，反正之后也不用
        pred_image = nib.Nifti1Image(pred_label, np.eye(4))
        pred_info = pd.DataFrame({
            "public_id": [img_id] * len(pred_index),  # 表示有多行
            "label_id": pred_index,
            "confidence": pred_proba,
            "label_code": pred_label_code
        })

    nib.save(pred_image, os.path.join(output_path, img_id + '.nii.gz'))
    return pred_info


def make_predictions(model, val_dataset,output_path, block_size, device):
    model.eval()
    for item in tqdm(val_dataset):
        # break
        pred_csv = execute_image(item,output_path=output_path, model=model, device=device, block_size=block_size)
        all_csv = pd.concat([all_csv, pred_csv], axis=0)
    all_csv.to_csv(os.path.join(output_path, 'pred_info.csv'), index=False)
