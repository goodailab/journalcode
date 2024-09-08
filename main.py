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
import PI
from torch.utils.data.dataset import Dataset
from utils import *

from torch.utils.tensorboard import SummaryWriter
import math
import argparse
from sklearn.metrics import r2_score
parser = argparse.ArgumentParser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser.add_argument('--sigma', type= float, default=1.0, help= "variance in n_z")
parser.add_argument('--lamb', type= float, default=0.5, help= "discrimintor wieght")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate for Adam optimizer")
parser.add_argument('--epochs',type=int, default=600, help="number of epochs for training")
parser.add_argument('--bs_tr', type=int, default=128, help="training batch size")
parser.add_argument('--bs_te', type = int, default=16, help="testing batch size")
parser.add_argument('--save', type=bool, default=False, help="save model weights during each epoch")
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--device', type= int, default=device, help="gpu device")
parser.add_argument('--data', type =int, default=1, help = "choose the data to run on")
parser.add_argument('--data_path', type=str, default='/home/ankita/Documents/PostDoc/LL/Code/bigdata/', help="simulated or hydra data")
parser.add_argument('--train', type=bool, default=True, help = "Train it")


parser.add_argument('--z_dim',type=int,default=16,help='before or after the vae parameterization')
parser.add_argument('--sc_wt', type =float, default=1e2, help="weight for scalar recon loss")
parser.add_argument('--sch', type=bool, default=False, help="save model weights during each epoch")
parser.add_argument('--wt', type =float, default=0.0, help="weight decay")
parser.add_argument('--scale', type =float, default=1.0, help="weight decay")
parser.add_argument('--model_type', type = str, default= 'wae', help="wae or poin")

args = parser.parse_args()

print(args.sch)

args.fold_name = '100k_mmd'+str(args.model_type)+str(args.sigma)+'sc_'+str(args.scale)+'bt_'+str(args.beta1)+'_'+str(args.wt)+'_'+str(args.z_dim)+'_'+str(args.lr)+'_lamb_'+str(args.lamb)+'_'+str(args.epochs)

torch.manual_seed(42)
np.random.seed(4321) 



print(args.fold_name)

def main_program(args):


    data = np.load('train_test_big.npz')
    y_img_test = data['y_img_test']
    y_sca_test = data['y_sca_test']
    y_img_train = data['y_img_train']
    y_sca_train = data['y_sca_train']

    trainset = CustomDataset(y_img_train,y_sca_train, res=False)
    testset = CustomDataset(y_img_test,y_sca_test, res =False)

    train_loader = DataLoader(dataset=trainset,batch_size= args.bs_tr,shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(dataset=testset,batch_size= args.bs_te,shuffle=False, num_workers=4)



    wae_encoder, wae_decoder= WAE_Encoder(args), WAE_Decoder(args)
    wae_encoder.to(device)
    wae_decoder.to(device)
    gd_sz = int(math.sqrt(args.bs_te))



    if args.train:

        criterion1 = nn.MSELoss()
        criterion2 = nn.MSELoss()

        enc_optim = torch.optim.Adam(wae_encoder.parameters(), lr = args.lr, betas= (args.beta1, 0.999), weight_decay = args.wt)
        dec_optim = torch.optim.Adam(wae_decoder.parameters(), lr = args.lr, betas= (args.beta1, 0.999), weight_decay = args.wt)
       







        if not os.path.exists('finalized/runs/' + args.fold_name):
            os.makedirs('finalized/runs/' + args.fold_name)
        writer = SummaryWriter('finalized/runs/'+ args.fold_name)



        for epoch in range(args.epochs):
            running_tr_img_recon_loss = 0.0
            running_tr_sclr_recon_loss = 0.0
            running_te_recon_loss = 0.0
            running_tr_recon_loss = 0.0
            running_tr_loss = 0.0

            running_te_norm_loss  = 0.0
            running_tr_norm_loss  = 0.0

            wae_encoder.train()
            wae_decoder.train()



            print('---------------------------', epoch)
            for i, (images, scalars) in enumerate(train_loader):
                args.tr =True

            
                images, scalars = images.to(device), scalars.to(device)
                wae_encoder.zero_grad()
                wae_decoder.zero_grad()
       
            


                z =  generate_prior(args.z_dim,images.size()[0], args.sigma, device, args.scale,  cons_type =args.model_type)
       

                z_hat = wae_encoder(images, scalars)

                mmd_loss = imq_kernel(z_hat, z, h_dim=args.z_dim)/images.size()[0]
                    
    
                znorm = torch.norm(z_hat, dim = 1)
                xi_hat, xs_hat = wae_decoder(z_hat)

                

                train_image_recon_loss = criterion1(xi_hat, images)
                train_scalar_recon_loss = criterion2(xs_hat, scalars)

                train_recon_loss = train_image_recon_loss + args.sc_wt * train_scalar_recon_loss


                d_loss = args.lamb *mmd_loss + train_recon_loss

              
            
                d_loss.backward()

                enc_optim.step()
                dec_optim.step()
               
                # print(train_image_recon_loss.data.item(),  train_scalar_recon_loss.data.item(),d_loss.data.item())
                running_tr_img_recon_loss += train_image_recon_loss.data.item()
                running_tr_sclr_recon_loss += train_scalar_recon_loss.data.item()
                running_tr_recon_loss += train_recon_loss.data.item()
                running_tr_loss += d_loss.data.item()
                running_tr_norm_loss += (torch.mean(znorm)).data.item()
                if i == 0:
                        
                    z_hat_tr = z_hat.data.cpu().numpy()
                else:
                    z_hat_tr = np.append(z_hat_tr, z_hat.data.cpu().numpy(), axis=0)
            wae_encoder.eval()
            wae_decoder.eval()
            # sca_true =[]
            # sca_pred = []
            for j, (images, scalars) in enumerate(test_loader):
                images, scalars = images.to(device), scalars.to(device)
                with torch.no_grad():
           
                    z_hat = wae_encoder(images, scalars)
                    znorm = torch.norm(z_hat, dim = 1)
                    xi_hat, xs_hat = wae_decoder(z_hat)
                    test_image_recon_loss = criterion1(xi_hat, images)
                    test_scalar_recon_loss = criterion2(xs_hat, scalars)
                    test_recon_loss = test_scalar_recon_loss + test_image_recon_loss
                    running_te_recon_loss += test_recon_loss.data.item()
                    running_te_norm_loss += (torch.mean(znorm)).data.item()
                    if j == 0:
                        sca_true = scalars.data.cpu().numpy().squeeze()
                        sca_pred = xs_hat.data.cpu().numpy().squeeze()
                        z_hat_all = z_hat.data.cpu().numpy()

                    else:
                        sca_true = np.append(sca_true, scalars.data.cpu().numpy().squeeze(), axis=0)
                        sca_pred = np.append(sca_pred, xs_hat.data.cpu().numpy().squeeze(), axis=0)
                        z_hat_all = np.append(z_hat_all, z_hat.data.cpu().numpy(), axis=0)




            # print stats after each epoch
            print("Epoch: [{}/{}], \tTrain Reconstruction Loss: {}, \tTrain Dis loss: {} \n"
                  "\t\t\tTest Reconstruction Loss:{}".format(epoch + 1, args.epochs,
                                                             running_tr_recon_loss/len(train_loader),running_tr_recon_loss/len(train_loader),
                                                             running_te_recon_loss/len(test_loader)))
            writer.add_scalar('recon_tr_img_loss',running_tr_img_recon_loss/len(train_loader), epoch)
            writer.add_scalar('recon_tr_sclr_loss',running_tr_sclr_recon_loss/len(train_loader), epoch)
            writer.add_scalar('tr_loss',running_tr_loss/len(train_loader), epoch)
            writer.add_scalar('recon_tr_recon_loss',running_tr_recon_loss/len(train_loader), epoch)
            writer.add_scalar('recon_te_loss',running_te_recon_loss/len(test_loader), epoch)
            writer.add_scalar('te_norm',running_te_norm_loss/len(test_loader), epoch)
            writer.add_scalar('tr_norm',running_tr_norm_loss/len(train_loader), epoch)
            writer.add_scalar('r2score_te',r2_score(sca_true,sca_pred))
            # grid_image = torchvision.utils.make_grid(xi_hat)

            if epoch % 5 == 0:
                fig_recon = fig_image_batch(xi_hat.data.cpu().numpy(), gd_size = gd_sz)
               
                fig_org = fig_image_batch(images.data.cpu().numpy(), gd_size = gd_sz)
                writer.add_figure('org', fig_org, global_step = epoch)
                writer.add_figure('recon', fig_recon, global_step = epoch)
                z =  generate_prior(args.z_dim,64, args.sigma, device, args.scale,  cons_type =args.model_type)

                xi_gen, xs_gen = wae_decoder(z)
                fig_gen = fig_image_batch(xi_gen.data.cpu().numpy(), gd_size =8)
                writer.add_figure('generated', fig_gen, global_step = epoch)

            if epoch ==(args.epochs-1):
                fig_recon = fig_image_batch(xi_hat.data.cpu().numpy(), gd_size = gd_sz) 
                fig_org = fig_image_batch(images.data.cpu().numpy(), gd_size = gd_sz)
                writer.add_figure('org', fig_org, global_step = epoch)
                writer.add_figure('recon', fig_recon, global_step = epoch)
                z =  generate_prior(args.z_dim,64, args.sigma, device, args.scale,  cons_type =args.model_type)

                xi_gen, xs_gen = wae_decoder(z)
                fig_gen = fig_image_batch(xi_gen.data.cpu().numpy(), gd_size = 8)
                writer.add_figure('generated_final', fig_gen, global_step = epoch)


        writer.close()
        save_path = 'checkpoints/{}_{}-epoch_{}.pth'
        torch.save(wae_encoder.state_dict(), save_path.format(args.fold_name,'encoder', epoch))
        torch.save(wae_decoder.state_dict(), save_path.format(args.fold_name,'decoder', epoch))

    else:
        print('train the model first')



main_program(args)

