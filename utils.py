
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
from scipy.stats import truncnorm
import random

from geomstats.geometry.hyperbolic import Hyperbolic




import torch

class CustomDataset(Dataset):
    def __init__(self, yimg, ysc, res= False):
        """
        Args:

            i
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        # self.to_tensor = transforms.ToTensor()
        # ead the csv file
        self.img_data = yimg
        self.scalar_data = ysc
        self.data_len =(self.img_data).shape[0]
        self.res =res

    def __getitem__(self, index):
        # Get image name from the pandas df
        img_as_img = self.img_data[index,]
        if self.res is True:
            img_as_img = np.resize(img_as_img, (224,224,4))
        img_as_tensor = img_as_img.transpose((2,0,1))
        # Get label(class) of the image based on the cropped pandas column
        single_image_sc = self.scalar_data[index,]

        return (img_as_tensor, single_image_sc)

    def __len__(self):
        return self.data_len


def generate_prior(n,batch_size, sig, device, scale, cons_type ='wae'):


    if cons_type =='wae':
        z = (torch.randn(batch_size, n) * sig).to(device)
    elif cons_type =='poin':
        poin = Hyperbolic(default_coords_type='ball', dim=n)
        z = ((torch.from_numpy(sig*poin.random_point(n_samples=batch_size,bound = scale))).type(torch.FloatTensor)).to(device)
        
    return z


def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats


class CustomDataset(Dataset):
    def __init__(self, yimg, ysc, res= False):

        self.img_data = yimg
        self.scalar_data = ysc
        self.data_len =(self.img_data).shape[0]
        self.res =res

    def __getitem__(self, index):
        # Get image name from the pandas df
        img_as_img = self.img_data[index,]
        if self.res is True:
            img_as_img = np.resize(img_as_img, (224,224,4))
        img_as_tensor = img_as_img.transpose((2,0,1))
        # Get label(class) of the image based on the cropped pandas column
        single_image_sc = self.scalar_data[index,]

        return (img_as_tensor, single_image_sc)

    def __len__(self):
        return self.data_len
    
# otheer helperr code for figure plottting
def plot_img(samples, grid_size, immax=None,immin=None):
    # plt.rcParams["figure.figsize"] = (10,10)
    IMAGE_SIZE = 64
    plt.close()

    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(grid_size, grid_size)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if immax is not None:
            plt.imshow(sample.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='winter',vmax=immax[i],vmin=immin[i])
    return fig


def fig_image_batch(imgs_batch, gd_size):
    imgs = imgs_batch.transpose((0,2,3,1))
    for k in range(4):
        fig = plot_img(imgs[:,:,:,k], grid_size=gd_size, immax=np.max(imgs[:,:,:,k].reshape(-1,4096),axis=1),
                   immin=np.min(imgs[:,:,:,k].reshape(-1,4096),axis=1))
        # plt.savefig('{}_img_{}_{}.png'.format(filename,str(k).zfill(3),str(k)),bbox_inches='tight')
        # plt.close()
    return fig

def process_image_batch(imgs):
    scales = [2.9258502e+01,8.5826596e+02,1.0004872e+05,4.8072070e+06]
    x0 = 0.029361539
    x1 = 0.016768705
    x2 = 0.0064973026
    x3 = 0.0028613657
    wt = [x0,x1,x2,x3]
    img_arr = imgs.transpose((0,2,3,1))
    img_ = np.copy(img_arr)
    for i in range(4):
        img_[:,:,:,i] /= scales[i]
        # img_[:,:,:,i] *= wt[i]
    return img_




def process_scalars(scalars):
    norma = [{ 'scale': 7.610738e+00,'bias': -4.075375e-01 },
             { 'scale': 1.459875e+00,'bias': -3.427656e+00 },
             { 'scale': 1.490713e+00,'bias': -3.495498e+00 },
             { 'scale': 4.375123e+01,'bias': -1.593477e+00 },
             { 'scale': 1.685576e-06,'bias': -5.330971e-01 },
             { 'scale': 1.430615e+00,'bias': -3.351173e+00 },
             { 'scale': 2.636422e-01,'bias': -9.762907e-01 },
             { 'scale': 7.154074e-18,'bias': -1.864709e-02 },
             { 'scale': 3.166824e-03,'bias': -1.864709e-02 },
             { 'scale': 2.102178e-02,'bias': -3.071955e-01 },
             { 'scale': 1.346439e+00,'bias': -3.118446e+00 },
             { 'scale': 2.419509e-01,'bias': -9.853402e-01 },
             { 'scale': 2.061877e-06,'bias': -5.213394e-01 },
             { 'scale': 1.392544e+00,'bias': -3.239921e+00 },
             { 'scale': 6.266253e-02,'bias': -1.384504e+00 }]
    sca = np.copy(scalars)
    for i in range(15):
        sca[:,i] = norma[i]['scale']*(scalars[:,i] - norma[i]['bias'])

    return sca


def imageTemp(pics):
    ### compute scientific constraint ###
    a         = np.array([[-0.02067386, -1.96963368],[-0.11814995, -1.5975888 ],[-0.12133077, -1.38709428]])
    ebin      = np.array([  12.,  20.,  35.,  50.])
    b         = np.sum(pics,axis=(0,1))
 
    lb        = np.log(1e-6+b)  # proportional to -h*nu/(k*T)
    slope     = (lb[1:]-lb[0:-1])/(ebin[1:]-ebin[0:-1])
    Ti        = np.exp(a[:,0]+a[:,1]*np.log(-slope))
    return np.mean(Ti), np.var(Ti)


def imageTemp_batch(batch):
    temp_m = []
    temp_v = []
    for img in batch:
        m,v = imageTemp(img)
        temp_m.append(m)
        temp_v.append(v)
    return np.array(temp_m),np.array(temp_v)