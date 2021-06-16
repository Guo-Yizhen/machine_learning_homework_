import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import nibabel as nib
from PIL import Image
import numpy as np
import tqdm
from skimage import measure
import random

import argparse

def adjust_border(centroid,i,bbox,data):
    if int(centroid[i]) - bbox / 2 <= 0:
        return 0,bbox
    if int(centroid[i]) + bbox / 2 >= data.shape[i]:
        return  (data.shape[i] - bbox), data.shape[i]
    return int(centroid[i]) - int(bbox / 2),int(centroid[i]) + int(bbox / 2)

def divide_data(bbox, label_list,data_list, root_data_path,root_label_path,label_save_path,data_save_path):

    for i in tqdm.tqdm(range(len(data_list))):
        data_file = data_list[i]
        label_file = label_list[i]
        data = nib.load(os.path.join(root_data_path, data_file)).get_data()
        label = nib.load(os.path.join(root_label_path, label_file)).get_data()
        property = measure.regionprops(label)
        for idx in range(len(property)):
            centroid = property[idx].centroid
            low_1,high_1= adjust_border( centroid, 0, bbox, data)
            low_2,high_2= adjust_border( centroid, 1, bbox, data)
            low_3,high_3= adjust_border( centroid, 2, bbox, data)
            box = data[low_1:high_1, low_2:high_2, low_3:high_3].astype(np.int16)
            target_box = label[low_1:high_1, low_2:high_2, low_3:high_3].astype(np.int16)
            precessed_data_file = os.path.join(data_save_path, data_file.split('-')[0] + '-' + str(idx + 1) + "-image")
            precessed_label_file = os.path.join(label_save_path, label_file.split('-')[0] + '-' + str(idx + 1) + "-label")
            np.save(precessed_data_file, box.reshape(1, *(box.shape)))
            np.save(precessed_label_file, target_box.reshape(1, *(box.shape)))

        num_empty = len(property)

        for idx in range(num_empty):
            low_1 = random.randint(0, data.shape[0] - bbox)
            high_1 = low_1 + bbox
            low_2 = random.randint(0, data.shape[1] - bbox)
            high_2 = low_2 + bbox
            low_3 = random.randint(0, data.shape[2] - bbox)
            high_3 = low_3 + bbox
            box = data[low_1:high_1, low_2:high_2, low_3:high_3].astype(np.int16)
            target_box = label[low_1:high_1, low_2:high_2, low_3:high_3].astype(np.int16)
            precessed_data_file = os.path.join(data_save_path, data_file.split('-')[0] + '-' + str(idx + 1 + num_empty) + "-image")
            precessed_label_file = os.path.join(label_save_path, label_file.split('-')[0] + '-' + str(idx + 1 + num_empty) + "-label")
            np.save(precessed_data_file, box.reshape(1, *(box.shape)))
            np.save(precessed_label_file, target_box.reshape(1, *(box.shape)))


def Divide_data():
    ROOT = os.path.join(os.getcwd(), 'dataset')
    process_path = os.path.join(ROOT, 'processed_data')
    train_data_path = os.path.join(process_path, "train_data")
    valid_data_path = os.path.join(process_path, 'val_data')
    test_data_path = os.path.join(ROOT, 'origin_data', 'test_data')
    train_label_path = os.path.join(process_path, 'train_label')
    valid_label_path = os.path.join(process_path, 'val_label')
    valid_test_like_path = os.path.join(ROOT, 'origin_data', 'val_data')
    origin_path = os.path.join(ROOT, 'origin_data')
    bbox=64
    

    #train data
    label_list = list(os.listdir(train_label_path))
    data_list = list(os.listdir(train_data_path))
    root_data_path = os.path.join(origin_path,  'train_data')
    root_label_path = os.path.join(origin_path, 'train_label')
    label_save_path = os.path.join(process_path,'train_label')
    data_save_path = os.path.join(process_path, 'train_data')
    divide_data(bbox,label_list,data_list, root_data_path,root_label_path,label_save_path,data_save_path)

    #val data
    label_list = list(os.listdir(valid_label_path))
    data_list = list(os.listdir(valid_data_path))
    root_data_path =  os.path.join(origin_path,  'val_data')
    root_label_path = os.path.join(origin_path,  'val_label')
    label_save_path = os.path.join(process_path, 'val_label')
    data_save_path = os.path.join(process_path,  'val_data')
    divide_data(bbox,label_list,data_list, root_data_path,root_label_path,label_save_path,data_save_path)

