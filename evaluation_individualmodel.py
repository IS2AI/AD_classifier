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
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import math
from classes import SingleImageTextReader,ResNet18

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0', help='choose gpus to train on')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs',type=int, default=200)
    parser.add_argument('--IMG_PATH', type=str, default='/datasets/MD_only/fold1/test')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--TESTING_PATH', type=str, default='/txt_files/MD_only/test_fold1_upd.txt')
    parser.add_argument('--model_path', type=str, default='/checkpoints/MD_only/model1_upd.pt')

    return parser.parse_args()


    
args = get_args()
    
TESTING_PATH = args.TESTING_PATH
IMG_PATH = args.IMG_PATH
batch_size = args.batch_size
epochs = args.batch_size
lr = args.lr    
gpu_ids = args.gpus
    
  
transforms_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
dset_test = AD_2DSlicesData(IMG_PATH, TESTING_PATH, transforms_test)

test_loader = DataLoader(dset_test,
                         batch_size = batch_size,
                         shuffle = False,
                         num_workers = 4,
                         drop_last=True
                         )
cuda='cuda:'+str(gpu_ids[0]) 
device=torch.device(cuda)
checkpoint = torch.load(args.model_path)
model = ResNet18(out_size=3)
model = nn.DataParallel(model, device_ids=gpu_ids)
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print(checkpoint['epoch'])
model.eval()
criterion = torch.nn.CrossEntropyLoss()
optimizer = eval("optim.Adam")(model.parameters(), lr)
ground_truths=[]
predictions=[]


for it, test_data in enumerate(test_loader):
    with torch.no_grad():
        imgs, labels = Variable(test_data['image']).to(device), Variable(test_data['label']).to(device) 
        img_input = imgs#.unsqueeze(1)
        integer_encoded = labels.data.cpu().numpy()
        ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
        for el in ground_truth.numpy():
            ground_truths.append(el)

    test_output = model(img_input)
    test_prob_predict = F.softmax(test_output, dim=1)
    _, predict = test_prob_predict.topk(1)
    predict=torch.transpose(predict, 0,1)
    predict2=predict.cpu().numpy()
    for el2 in predict2:
        for el3 in el2:
            predictions.append(el3)
            
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ground_truths, predictions))
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(ground_truths, predictions))
                                              