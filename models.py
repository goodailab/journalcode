

import torch
import torch.nn as nn
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
from torch.utils.data.dataset import Dataset



class WAE_Encoder(nn.Module):
    def __init__(self, args):
        super(WAE_Encoder, self).__init__()
        self.z_dim =args.z_dim
        # convolutional filters, work excellent with image data
        self.encode = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 4, 2, padding =1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 4, 2, padding = 1, bias=True),
            nn.ReLU(True)
        )
        self.d1 = nn.Linear(1024+15,self.z_dim)
    def constrained(self, z):
        # if self.model_type == 'sph':
        nm = (torch.norm(z, dim = 1).view(z.shape[0],1))
        return torch.div(z, nm)
    def forward(self, x, xs):
        x = self.encode(x)
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        # x = self.constrained(x)
        xcat = torch.cat((x,xs),1)
        xout = self.d1(xcat)
        return xout





class WAE_Decoder(nn.Module):
    def __init__(self, args):
        super(WAE_Decoder, self).__init__()

        # first layer is fully connected
        self.z_dim = args.z_dim
        self.d2 = nn.Sequential(
            nn.Linear(self.z_dim,1024),
            nn.ReLU(True)
        )
        self.d3 = nn.Linear(1024,15)
        # deconvolutional filters, essentially the inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16,32, 4,2, padding =1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,16, 4,2, padding =1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 4, 4,2, padding =1, bias=True)
        )
    def constrained(self, z):
        # if self.model_type == 'sph':
        nm = (torch.norm(z, dim = 1).view(z.shape[0],1))
        return torch.div(z, nm)

    def forward(self, x):
        x = self.d2(x)
        # x = self.constrained(x)
        xs = self.d3(x)
        x = x.view(-1, 16, 8,8)
        x = self.deconv(x)
        return x, xs
