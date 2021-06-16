import os
import nibabel as nib
import math
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import random
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

ROOT = os.path.join(os.getcwd(), 'dataset')
process_path = os.path.join(ROOT, 'processed_data')
train_data_path = os.path.join(process_path, "train_data")
valid_data_path = os.path.join(process_path, 'val_data')
test_data_path = os.path.join(ROOT, 'origin_data', 'test_data')
train_label_path = os.path.join(process_path, 'train_label')
valid_label_path = os.path.join(process_path, 'val_label')
valid_test_like_path = os.path.join(ROOT, 'origin_data', 'val_data')

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(3)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class FracRibtrainDataSet(Dataset):
    def __init__(self, args, set="train"):
        self.transform = torch.from_numpy
        self.label_transform = torch.from_numpy
        self.flip1 = RandomFlip_UD(prob=0.3)
        self.flip2 = RandomFlip_LR(prob=0.3)

        if set=='train':
            self.data_list = list(os.listdir(train_data_path))
            self.label_list = list(os.listdir(train_label_path))
            self.root_data_path = train_data_path
            self.root_label_path = train_label_path
        elif set == "val":
            self.data_list = list(os.listdir(valid_label_path))
            self.label_list =  list(os.listdir(valid_label_path))
            self.root_data_path = valid_data_path
            self.root_label_path = valid_label_path
        self.set = set

    def __getitem__(self, index):
        data_file = os.path.join(self.root_data_path, self.data_list[index])
        img = np.load(data_file, allow_pickle=True)
        # print(img.shape)

        img[img >= 500] = 500
        img[img <= -500] = -500
        img = np.array(img, dtype=float)
        img /= 500

        if self.transform is not None:
            img = self.transform(img)
        label_file = os.path.join(self.root_label_path, self.label_list[index])
        target = np.load(label_file)
        target[target != 0] = 1  # 二值化
        if self.label_transform is not None:
            target = self.label_transform(target)

        # img, target = self.flip1(self.flip2(img, target))
        img, target = self.flip1(img, target)
        img, target = self.flip2(img, target)

        return img, target

    def __len__(self):
        return len(self.data_list)


class FracRibTestDataSet(Dataset):
    def __init__(self,args, set="val"):
        self.transform = torch.from_numpy
        self.label_transform = torch.from_numpy
        self.block_size = args.block_size

        if set == "val":
            self.data_list = os.listdir(valid_test_like_path)
            self.root_data_path = valid_test_like_path
        elif set == 'test':
            self.data_list = list(os.listdir(test_data_path))
            self.root_data_path = test_data_path
        else:
            print('only val and test set are supported for RibTest')
            exit(1)
        self.set = set

    def __getitem__(self, index):
        data_file = os.path.join(self.root_data_path, self.data_list[index])
        img = nib.load(data_file).get_data()
        img[img >= 500] = 500
        img[img <= -500] = -500
        img = np.array(img, dtype=float)
        img /= 500
        block_size = self.block_size
        img_id = self.data_list[index].split('-')[0]

        # img_list = []

        l_iter = math.ceil(img.shape[0] / (block_size / 2)) - 1
        w_iter = math.ceil(img.shape[1] / (block_size / 2)) - 1
        d_iter = math.ceil(img.shape[2] / (block_size / 2)) - 1

        return img, img_id, img.shape

    def __len__(self):
        return len(self.data_list)


