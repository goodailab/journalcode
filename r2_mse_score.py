import os
import sys 
# from model_flex import *
# from model_dec_mod import *
from models import *
# sys.path.append('../')
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import manifold
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import PIL
import torch.nn.functional as F
sys.path.append("/home/ankita/Code/LL")
from torch.utils.data.dataset import Dataset
from utils import CustomDataset
from scipy.optimize import curve_fit


from utils import process_image_batch, process_scalars,imageTemp_batch
import math
import argparse
from sklearn.metrics import r2_score
parser = argparse.ArgumentParser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
parser.add_argument('--bs_te', type = int, default=16, help="testing batch size")
parser.add_argument('--save', type=bool, default=False, help="save model weights during each epoch")
parser.add_argument('--device', type= int, default=device, help="gpu device")
parser.add_argument('--epochs',type=int, default=600, help="number of epochs for training")
parser.add_argument('--fold_name', type = str, default= '1', help="wae or poin")

args = parser.parse_args()
def linear_func(x,a,b):

    return a*x+b

def main_program(args):
    data = np.load('train_test_big.npz')
    y_img_test = data['y_img_test']
    y_sca_test = data['y_sca_test']




    testset = CustomDataset(y_img_test,y_sca_test)

    test_loader = DataLoader(dataset=testset,batch_size= args.bs_te,shuffle=True)

    wae_encoder, wae_decoder = WAE_Encoder(args), WAE_Decoder(args)
    wae_encoder.to(device)
    wae_decoder.to(device)

    enc_checkpoint = torch.load('finalized/checkpoints/'+args.fold_name+'_encoder-epoch_{}.pth'.format(args.epochs-1))
    wae_encoder.load_state_dict(enc_checkpoint)

    dec_checkpoint = torch.load('finalized/checkpoints/'+args.fold_name+'_decoder-epoch_{}.pth'.format(args.epochs-1))
    wae_decoder.load_state_dict(dec_checkpoint)

    wae_encoder.eval()
    wae_decoder.eval()

    criterion1 = nn.MSELoss()
    running_tr_recon_loss =0.0
# #############################
    for j, (images, scalars) in enumerate(test_loader):

        images, scalars = images.to(device), scalars.to(device)

        with torch.no_grad():
            
            z_hat= wae_encoder(images,scalars)
            xi_hat, xs_hat = wae_decoder(z_hat)
            imgs_gt = process_image_batch(images.data.cpu().numpy())
            sca_gt = process_scalars(scalars.data.cpu().numpy())
            mu_gt,va_gt = imageTemp_batch(imgs_gt)
            imgs_pred = process_image_batch(xi_hat.data.cpu().numpy())
            sca_pred = process_scalars(xs_hat.data.cpu().numpy())
            mu_pred,va_pred = imageTemp_batch(imgs_pred)
            mse= criterion1(xi_hat, images)

            if j ==0:
                sca_gt_all = sca_gt.copy()
                mu_gt_all = mu_gt.copy()
                # z_hat = wae_encoder(images, scalars)
                sca_pred_all = sca_pred.copy()
                mu_pred_all = mu_pred.copy()
            else:
                sca_gt_all = np.append(sca_gt_all, sca_gt, axis=0)
                mu_gt_all = np.append(mu_gt_all, mu_gt, axis=0)
                sca_pred_all = np.append(sca_pred_all, sca_pred, axis=0)
                mu_pred_all = np.append(mu_pred_all, mu_pred, axis=0)
        running_tr_recon_loss += mse.data.item()
    r2 = r2_score(sca_gt_all,sca_pred_all)
    mse_loss = running_tr_recon_loss/len(test_loader)
    print('r2 and mse scores on test data are:')
    print(r2)
    print(mse_loss)

        
main_program(args)