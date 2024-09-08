import os
import sys 
from models import *

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


import PIL
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from utils import CustomDataset
from scipy.optimize import curve_fit

from torch.utils.tensorboard import SummaryWriter
from utils import generate_prior, process_image_batch, process_scalars,imageTemp_batch
import argparse
from sklearn.metrics import r2_score
from utils import coverage

parser = argparse.ArgumentParser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser.add_argument('--sigma', type= float, default=1.0, help= "variance in n_z")
parser.add_argument('--epochs',type=int, default=600, help="number of epochs for training")
parser.add_argument('--bs_tr', type=int, default=128, help="training batch size")
parser.add_argument('--device', type= int, default=device, help="gpu device")
parser.add_argument('--z_dim',type=int,default=16,help='before or after the vae parameterization')
parser.add_argument('--scale', type =float, default=1.0, help="weight decay")
parser.add_argument('--sigma', type= float, default=1.0, help= "variance in n_z")
parser.add_argument('--fold_name', type = str, default= '1', help="wae or poin")

args = parser.parse_args()
def linear_func(x,a,b):

    return a*x+b


def main_program(args):
    data = np.load('train_test_big.npz')
    y_img_train = data['y_img_train']
    y_sca_train = data['y_sca_train']

    trainset = CustomDataset(y_img_train,y_sca_train, res=False)

    train_loader = DataLoader(dataset=trainset,batch_size= args.bs_tr,shuffle=True, drop_last=True)

    wae_encoder, wae_decoder = WAE_Encoder(args), WAE_Decoder(args)
    wae_encoder.to(device)
    wae_decoder.to(device)

    enc_checkpoint = torch.load('finalized/checkpoints/'+args.fold_name+'_encoder-epoch_{}.pth'.format(args.epochs-1))
    wae_encoder.load_state_dict(enc_checkpoint)

    dec_checkpoint = torch.load('finalized/checkpoints/'+args.fold_name+'_decoder-epoch_{}.pth'.format(args.epochs-1))
    wae_decoder.load_state_dict(dec_checkpoint)

    wae_encoder.eval()
    wae_decoder.eval()


# #############################
    for j, (images, scalars) in enumerate(train_loader):

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

            if j ==0:
                sca_gt_all = sca_gt.copy()
                mu_gt_all = mu_gt.copy()
                # z_hat = wae_encoder(images, scalars)
                sca_pred_allgt = sca_pred.copy()
                mu_pred_allgt = mu_pred.copy()
         

            else:
                sca_gt_all = np.append(sca_gt_all, sca_gt, axis=0)
                mu_gt_all = np.append(mu_gt_all, mu_gt, axis=0)
                sca_pred_allgt = np.append(sca_pred_allgt, sca_pred, axis=0)
                mu_pred_allgt = np.append(mu_pred_allgt, mu_pred, axis=0)

    # fit a line on training data
    popt, pcov = curve_fit(linear_func, sca_gt_all[:,10],mu_gt_all)
    err_gt = np.abs(linear_func(sca_pred_allgt[:,10], *popt) - mu_pred_allgt)
  

    tot_all =  len(np.where(err_gt <= (np.mean(err_gt)+np.std(err_gt)))[0])


# #######################'Generate samples for evalauting validity and coverage
    ind_good= 0
    ind_bad = 0
    tot = 0
    for i in range(100):
        with torch.no_grad():
            tot +=1024

            z =  generate_prior(args.z_dim,1024, args.sigma, device, args.scale,  cons_type =args.model_type)

            xi_hat, xs_hat = wae_decoder(z)



            imgs_pred = process_image_batch(xi_hat.data.cpu().numpy())
            sca_pred = process_scalars(xs_hat.data.cpu().numpy())
            mu_pred,va_pred = imageTemp_batch(imgs_pred)
            err_vec = np.square(linear_func(sca_pred[:,10], *popt) - mu_pred)

            ind = np.where(err_vec <= 0.001)[0]
            ind_g = np.where(err_vec > 0.001)[0]
   
            if i ==0:
                sca_pred_all = sca_pred[ind,:].copy()
                mu_pred_all = mu_pred[ind].copy()
                
            else:
                sca_pred_all = np.append(sca_pred_all, sca_pred[ind,:], axis=0)
                mu_pred_all = np.append(mu_pred_all, mu_pred[ind], axis=0)
            ind_good += len(ind)
            ind_bad += len(ind_g)
 
    cov_x = coverage(sca_gt_all[:,10], sca_pred_all[:,10])
    cov_y = coverage(mu_gt_all, mu_pred_all)
  
   
    f =open("fit_metric_paper_results.txt", "a+")
    f.write("valid generated:  %.2f, %.2f\n "%(ind_good/tot, ind_bad/tot))
    f.write("Coverage x and y: %.2f, %.2f\n"%(cov_x, cov_y))
    f.write("tot: %.3f"%(tot_all/len(err_gt)))
    f.close()
main_program(args)