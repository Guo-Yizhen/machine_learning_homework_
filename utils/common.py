import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import torch, random


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.75 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

