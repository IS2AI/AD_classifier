import sys
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image

import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import random
import torch.optim as optim 

import nibabel as nib
import os
from skimage.transform import resize 
import random
import torch.nn as nn
import math

class ResNet18(nn.Module):

    def __init__(self, out_size):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )

    def forward(self, x):
        x = self.resnet18(x)
        return x
    
    
    

class SingleImageTextReader(Dataset):
    
    def __init__(self, root_dir, data_file, transform=None, slice = slice):
  
        self.root_dir = root_dir
        self.data_file = data_file
        self.transform = transform
    
    def __len__(self):
        return sum(1 for line in open(self.data_file))
    
    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        lst = lines[idx].split()
        img_name = lst[0]
        img_label = lst[1]
        image_path = os.path.join(self.root_dir, img_name)
        image = Image.open(image_path)
        if img_label == 'Normal':
            label = 0
        elif img_label == 'AD':
            label = 2
           
        elif img_label == 'MCI':
            label = 1
           
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        
        return sample
    
class AgnosticTextReader(Dataset):
    
    def __init__(self, root_dir_1, root_dir_2, root_dir_3, data_file, transform=None, slice = slice):

        self.root_dir_1 = root_dir_1
        self.root_dir_2 = root_dir_2
        self.root_dir_3 = root_dir_3
        self.data_file = data_file
        self.transform = transform
    
    def __len__(self):
        return sum(1 for line in open(self.data_file))
    
    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        lst = lines[idx].split()
        num_images=len(lst)-1
        if num_images == 3:
            img_name_1 = lst[0]
            img_name_2 = lst[1]
            img_name_3 = lst[2]
            img_label = lst[3]
            image_path1 = os.path.join(self.root_dir_1, img_name_1)
            image_path2 = os.path.join(self.root_dir_2, img_name_2)
            image_path3 = os.path.join(self.root_dir_3, img_name_3)
            image1 = Image.open(image_path1)
            image2 = Image.open(image_path2)
            image3 = Image.open(image_path3)

           
        elif num_images == 2:
            img_name_1 = lst[0]
            img_name_2 = lst[1]
            img_label = lst[2]
            image_path1 = os.path.join(self.root_dir_1, img_name_1)
            image_path2 = os.path.join(self.root_dir_2, img_name_2)
            image1 = Image.open(image_path1)         
            image2 = Image.open(image_path2)
            image3 = Image.new("RGB", (224, 224), (255, 255, 255))

          
        elif num_images == 1:
            img_name = lst[0]
            img_label = lst[1]
            if img_name[0]=='r':
                image_path = os.path.join(self.root_dir_3, img_name)
                image1 = Image.new("RGB", (224, 224), (255, 255, 255))
                image2 = Image.new("RGB", (224, 224), (255, 255, 255))
                image3 = Image.open(image_path)         

            elif img_name[0]=='F':
                image_path = os.path.join(self.root_dir_1, img_name)
                image1 =  Image.open(image_path)       
                image2 = Image.new("RGB", (224, 224), (255, 255, 255))
                image3 =Image.new("RGB", (224, 224), (255, 255, 255))
            elif img_name[0]=='M':
                image_path = os.path.join(self.root_dir_2, img_name)
                image1 = Image.new("RGB", (224, 224), (255, 255, 255))    
                image2 = Image.open(image_path) 
                image3 = Image.new("RGB", (224, 224), (255, 255, 255))
                    
                
                
        if img_label == 'Normal':
            label = 0
        elif img_label == 'AD':
            label = 2
        elif img_label == 'MCI':
            label = 1
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)

        
        sample = {'t1w': image3,'FA':image1, 'MD':image2, 'label': label}
        return sample
class ThreeInputTextReader(Dataset):
    """labeled Faces in the Wild dataset."""
    
    def __init__(self, root_dir_1, root_dir_2, root_dir_3, data_file, transform=None, slice = slice):
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.root_dir_1 = root_dir_1
        self.root_dir_2 = root_dir_2
        self.root_dir_3 = root_dir_3
        self.data_file = data_file
        self.transform = transform
    
    def __len__(self):
        return sum(1 for line in open(self.data_file))
    
    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        lst = lines[idx].split()
        img_name_1 = lst[0]
        img_name_2 = lst[1]
        img_name_3 = lst[2]

        img_label = lst[3]
        image_path1 = os.path.join(self.root_dir_1, img_name_1)
        image_path2 = os.path.join(self.root_dir_2, img_name_2)
        image_path3 = os.path.join(self.root_dir_3, img_name_3)
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)
        image3 = Image.open(image_path3)

        if img_label == 'Normal':
            label = 0
        elif img_label == 'AD':
            label = 2
           
        elif img_label == 'MCI':
            label = 1
      #  elif img_label=='Demented':
           # label = 3

    
        #image = Image.fromarray(image.astype(np.uint8), 'RGB')
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)

        sample = {'t1w': image3,'FA':image1, 'MD':image2, 'label': label}
        
        return sample
    
    
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB,modelC, nb_classes=3):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        # Remove last linear layer
        self.modelA.module.resnet18.fc = nn.Identity()
        self.modelB.module.resnet18.fc = nn.Identity()
        self.modelC.module.resnet18.fc = nn.Identity()
        # Create new classifier
        self.classifier = nn.Linear(512+512+512, nb_classes)
        
    def forward(self, x1,x2,x3):
        x1 = self.modelA(x1) 
        x2 = self.modelB(x2)
        x3 = self.modelC(x3)
        x = torch.cat((x1, x2, x3), dim=1)   
        x = self.classifier(F.relu(x))
        return x
    