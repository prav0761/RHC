#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
from time import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from zipfile import ZipFile
import numpy as np
from time import time
from torchvision import datasets
from torchvision import transforms
import pandas as pd
import numpy as np
import zipfile
import os
from torchvision.io import read_image


# In[4]:


def collect_data(datatype):
    image_dir=r"F:\New folder (2)\RHC\img"+'\\'+datatype
    images_all=[]
    density_map_all=[]
    for i in os.listdir(image_dir):
        tsr_img1 = torchvision.io.read_image(image_dir+'\\'+i)
        images_all.append(tsr_img1)
        anno_dir=r"F:\New folder (2)\RHC\anno\all\den"+'\\'+datatype
        d1=pd.read_csv(anno_dir+'\\'+i[:-3]+'csv',header=None)
        density_map_all.append(torch.tensor(np.array(d1)))
    return images_all,density_map_all


# In[5]:


class MyDataset(Dataset):
    def __init__(self, data, targets,transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
    
    def __len__(self):
        return len(self.data)


# In[6]:


def check_image_and_label(data,index):
    x,y=data[index]
    plt.imshow(x.permute(1,2,0))
    plt.show()
    plt.imshow(y)
    plt.show