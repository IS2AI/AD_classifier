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
from classes import ResNet18,SingleImageTextReader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0', help='choose gpus to train on')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs',type=int, default=200)
    parser.add_argument('--IMG_PATH_1', type=str, default='/dataset/MD_only/fold1/train')
    parser.add_argument('--IMG_PATH_2', type=str, default='/dataset/MD_only/fold1/test')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--TRAINING_PATH', type=str, default='/txt_files/MD_only/train_fold1_upd.txt')
    parser.add_argument('--TESTING_PATH', type=str, default='/txt_files/MD_only/test_fold1_upd.txt')
    parser.add_argument('--model_path', type=str, default='/checkpoints/MD_only/model1_upd.pt')

    return parser.parse_args()


    
args = get_args()
    
# Path configuration
TRAINING_PATH = args.TRAINING_PATH
TESTING_PATH = args.TESTING_PATH
IMG_PATH_1 = args.IMG_PATH_1
IMG_PATH_2 = args.IMG_PATH_2
batch_size = args.batch_size
epochs = args.batch_size
lr = args.lr


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
dset_train = SingleImageTextReader(IMG_PATH_1, TRAINING_PATH, transforms_train)
dset_test = SingleImageTextReader(IMG_PATH_2, TESTING_PATH, transforms_test)
print(len(dset_train))

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




# Training process
model = ResNet18(out_size=3)
#gpu ids are set here
gpu_ids=[13,15]
cuda='cuda:'+str(gpu_ids[0]) 
model = nn.DataParallel(model, device_ids=gpu_ids)
device=torch.device(cuda)

model.to(device) 

criterion = torch.nn.CrossEntropyLoss()
optimizer = eval("optim.Adam")(model.parameters(), lr)
                                                   
last_dev_avg_loss = float("inf")
best_accuracy = float("-inf")

# main training loop


for epoch_i in range(epochs):
    print("At {0}-th epoch.".format(epoch_i))
    train_loss = 0.0
    correct_cnt = 0.0
    model.train()
    for it, train_data in enumerate(train_loader):
        #data_dic = train_data

        #if use_cuda:
        

        image_input, labels = Variable(train_data['image']).to(device),Variable(train_data['label']).to(device) 


        integer_encoded = labels.data.cpu().numpy()
        # target should be LongTensor in loss function
        ground_truth = Variable(torch.from_numpy(integer_encoded)).long().to(device)
        #if use_cuda:
         #   ground_truth = ground_truth.cuda()
        train_output = model(image_input)
        train_prob_predict = F.softmax(train_output, dim=1)
        _, predict = train_prob_predict.topk(1)
        loss = criterion(train_output, ground_truth)

        train_loss += loss
        correct_this_batch = (predict.squeeze(1) == ground_truth).sum()
        correct_cnt += correct_this_batch
        accuracy = float(correct_this_batch) / len(ground_truth)
        print("batch {0} training loss is : {1:.5f}".format(it, loss.data))
        print("batch {0} training accuracy is : {1:.5f}".format(it, accuracy))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_avg_loss = train_loss / (len(dset_train) / batch_size)
    train_avg_acu = float(correct_cnt) / len(dset_train)
    
    print("Average training loss is {0:.5f} at the end of epoch {1}".format(train_avg_loss.data, epoch_i))
    print("Average training accuracy is {0:.5f} at the end of epoch {1}".format(train_avg_acu, epoch_i))
    

    # validation -- this is a crude esitmation because there might be some paddings at the end
    dev_loss = 0.0
    correct_cnt = 0.0
    model.eval()
    for it, test_data in enumerate(test_loader):
        with torch.no_grad():
            img_input, labels = Variable(test_data['image']).to(device), Variable(test_data['label']).to(device) 
            integer_encoded = labels.data.cpu().numpy()
            ground_truth = Variable(torch.from_numpy(integer_encoded)).long().to(device)
    
        test_output = model( img_input)
        test_prob_predict = F.softmax(test_output, dim=1)
        _, predict = test_prob_predict.topk(1)
        loss = criterion(test_output, ground_truth)
        dev_loss += loss
        correct_this_batch = (predict.squeeze(1) == ground_truth).sum()
        correct_cnt += (predict.squeeze(1) == ground_truth).sum()
        accuracy = float(correct_this_batch) / len(ground_truth)
        print("batch {0} dev loss is : {1:.5f}".format(it, loss.data))
        print("batch {0} dev accuracy is : {1:.5f}".format(it, accuracy))

    dev_avg_loss = dev_loss / (len(dset_test) / batch_size)
    dev_avg_acu = float(correct_cnt) / len(dset_test)
    
    print("Average validation loss is {0:.5f} at the end of epoch {1}".format(dev_avg_loss.data, epoch_i))
    print("Average validation accuracy is {0:.5f} at the end of epoch {1}".format(dev_avg_acu, epoch_i))
    #model is saved with respect to the average accuracy value 
    if dev_avg_acu>best_accuracy:  
        best_accuracy=dev_avg_acu
        torch.save({ 'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, args.model_path) # official recommended

    last_dev_avg_loss = dev_avg_loss

