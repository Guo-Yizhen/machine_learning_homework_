import sys
from dataset import FracRibtrainDataSet, FracRibTestDataSet
import nibabel as nib
import argparse
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from functools import partial
from my_Unet import Unet
from divide_data_ import Divide_data
from utils import logger,metrics, common, loss
import os
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from prediction import make_predictions
import pandas as pd

def adjust_learning_rate(optimizer, epoch, args):
   
    lr = args.lr * (0.75 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def val(model, val_loader, loss_func):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(2)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.sigmoid(output)
            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))
            output[output>=0.5]=1.0
            output[output<0.5]=0.0
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice': val_dice.avg[1]})
    return val_log


def train(model, train_loader, optimizer, loss_func,epoch):
    
    print("Epoch:{}       lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

    model.train()

    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        data, target = data.float(), target.long()  # both: (B, Channel, H, L, W);
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        output = torch.sigmoid(output)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()



def epoch_loop(model,train_loader,val_loader,loss,optimizer,save_path):
    best = [0, 0]  # 初始化最优模型的epoch和performance

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        train(model, train_loader, optimizer, loss, epoch)
        
        val_log = val(model, val_loader, loss)

        if val_log['Val_dice'] > best[1]:
            print('Saving best model')
            for file in os.listdir(save_path):  # 删掉上一次的best model
                if file.startswith('best_model'):
                    os.remove(os.path.join(save_path, file))
                    break
            torch.save(os.path.join(save_path, f'best_model_{epoch}.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper-parameters management')
    parser.add_argument('--upper', type=int, default=500)
    parser.add_argument('--lower', type=int, default=-500)
    parser.add_argument('--save', default='Unet-6-6-lr5e-2')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.05)

    args = parser.parse_args()
    save_path = os.path.join(os.getcwd(), 'model_saved')
    output_path_t=os.path.join(os.getcwd(), 'pred_output_test')
    output_path_v=os.path.join(os.getcwd(), 'pred_output_val')
    
    device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')

    #prepare the data
    Divide_data()


    #train the model
    train_loader = DataLoader(dataset=FracRibtrainDataSet(args, set='train'), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=FracRibtrainDataSet(args, set='val'), batch_size=4, shuffle=True)

    model = Unet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss = loss.DiceLoss()

    
    epoch_loop(model,train_loader,val_loader,loss,optimizer,save_path)
 
    # predicte
    test_dataset=FracRibTestDataSet(args, set='test')
    val_dataset=FracRibTestDataSet(args, set='val')
    make_predictions(model, test_dataset, output_path_t, 64, device)
    make_predictions(model, val_dataset, output_path_v, 64, device)


