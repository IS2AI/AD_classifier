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
import pandas as pd
from classes import AgnosticTextReader,ResNet18,MyEnsemble

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0', help='choose gpus to train on')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs',type=int, default=200)
    parser.add_argument('--IMG_PATH_1', type=str, default='/dataset/agnostic/fold1/train/FA')
    parser.add_argument('--IMG_PATH_2', type=str, default='/dataset/agnostic/fold1/train/MD')
    parser.add_argument('--IMG_PATH_3', type=str, default='/dataset/agnostic/fold1/train/T1w')
    parser.add_argument('--IMG_PATH_4', type=str, default='/dataset/agnostic/fold1/test/FA')
    parser.add_argument('--IMG_PATH_5', type=str, default='/dataset/agnostic/fold1/test/MD')
    parser.add_argument('--IMG_PATH_6', type=str, default='/dataset/agnostic/fold1/test/T1w')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--TRAINING_PATH', type=str, default='/txt_files/agnostic/train_fold1.txt')
    parser.add_argument('--TESTING_PATH', type=str, default='/txt_files/agnostic/test_fold1_upd.txt')
    parser.add_argument('--model_path_FA', type=str, default='/checkpoints/FA_only/model1_upd.pt')
    parser.add_argument('--model_path_MD', type=str, default='/checkpoints/MD_only/model1_upd.pt')
    parser.add_argument('--model_path_T1W', type=str, default='/checkpoints/T1w_only/model1_upd.pt')
    parser.add_argument('--model_path', type=str, default='/checkpoints/agnostic/model1_upd.pt')

 


    return parser.parse_args()


    
args = get_args()
    
TRAINING_PATH = args.TRAINING_PATH
TESTING_PATH = args.TESTING_PATH
IMG_PATH_1 = args.IMG_PATH_1
IMG_PATH_2 = args.IMG_PATH_2
IMG_PATH_3 = args.IMG_PATH_3
IMG_PATH_4 = args.IMG_PATH_4
IMG_PATH_5 = args.IMG_PATH_5
IMG_PATH_6 = args.IMG_PATH_6
model_path_T1W=args.model_path_T1W
model_path_FA=args.model_path_FA
model_path_MD=args.model_path_MD

batch_size = args.batch_size
epochs = args.batch_size
lr = args.lr
gpu_ids=args.gpus





transforms_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

transforms_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
dset_train = AgnosticTextReader(IMG_PATH_1, IMG_PATH_2,IMG_PATH_3,TRAINING_PATH, transforms_train)
dset_test = AgnosticTextReader(IMG_PATH_4, IMG_PATH_5,IMG_PATH_6, TESTING_PATH, transforms_test)


# Use argument load to distinguish training and testing
train_loader = DataLoader(dset_train,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = 4,
                          drop_last = True
                         )

test_loader = DataLoader(dset_test,
                         batch_size = batch_size,
                         shuffle = False,
                         num_workers = 4,
                         drop_last=True
                         )

odelA = ResNet18(out_size=3)
modelB = ResNet18(out_size=3)
modelC = ResNet18(out_size=3)


optimizerA = eval("optim.Adam")(modelA.parameters(), 1e-4)      
optimizerB = eval("optim.Adam")(modelB.parameters(), 1e-4)
optimizerC = eval("optim.Adam")(modelC.parameters(), 1e-4)      

# Freeze these models
for param in modelA.parameters():
    param.requires_grad = False

for param in modelB.parameters():
    param.requires_grad = False

    
for param in modelC.parameters():
    param.requires_grad = False
    
cuda='cuda:'+str(gpu_ids[0]) 
device=torch.device(cuda)
#cuda='cuda:0'
modelA = nn.DataParallel(modelA, device_ids=gpu_ids)
modelB = nn.DataParallel(modelB, device_ids=gpu_ids)
modelC = nn.DataParallel(modelC, device_ids=gpu_ids)


modelA.to(device)    
modelB.to(device)    
modelC.to(device)    

checkpointA = torch.load(model_path_T1W)
modelA.load_state_dict(checkpointA['model_state_dict'])
optimizerA.load_state_dict(checkpointA['optimizer_state_dict'])

checkpointB = torch.load(model_path_FA)
modelB.load_state_dict(checkpointB['model_state_dict'])
optimizerB.load_state_dict(checkpointB['optimizer_state_dict'])

checkpointC = torch.load(model_path_MD)
modelC.load_state_dict(checkpointC['model_state_dict'])
optimizerC.load_state_dict(checkpointC['optimizer_state_dict'])
                
# Create ensemble model
model = MyEnsemble(modelA, modelB, modelC)
model = nn.DataParallel(model, device_ids=gpu_ids)
model.to(device)    
criterion = torch.nn.CrossEntropyLoss()
optimizer = eval("optim.Adam")(model.parameters(), lr)
model = MyEnsemble(modelA, modelB, modelC)
model = nn.DataParallel(model, device_ids=gpu_ids)
model.to(device)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

ground_truths=[]
predictions=[]
for it, test_data in enumerate(test_loader):
    with torch.no_grad():
        t1w,FA, MD, labels = Variable(test_data['t1w']).to(device),Variable(test_data['FA']).to(device),Variable(test_data['MD']).to(device), Variable(test_data['label']).to(device) 
        integer_encoded = labels.data.cpu().numpy()
        ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
        for el in ground_truth.numpy():
            ground_truths.append(el)

    test_output = model(t1w,FA,MD)
    test_prob_predict = F.softmax(test_output, dim=1)
    _, predict = test_prob_predict.topk(1)
    predict=torch.transpose(predict, 0,1)
    predict2=predict.cpu().numpy()
    for el2 in predict2:
        for el3 in el2:
            predictions.append(el3)
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
print(confusion_matrix(ground_truths, predictions))
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(ground_truths, predictions))