# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:54:13 2021

@author: LEGION
"""
import torch.nn as nn
import torch.nn.functional as F

class resnetblock1(nn.Module):
    def __init__(self,in_channel):
        super(resnetblock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,in_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel,in_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm2d(in_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(x+y)
        return y

class resnetblock2(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(resnetblock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size = 1,stride = 2,bias = False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channel,out_channel,kernel_size = 3,stride = 2 ,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn3 = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        y = self.conv2(x)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y+z)
        return y

class resnetblock3(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(resnetblock3,self).__init__()
        F1, F2, F3 = filters
        self.conv1 = nn.Conv2d(in_channel,F1,1,stride=s, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1,F2,f,stride=1, padding=True, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(F3)
        self.shortcut = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.bns = nn.BatchNorm2d(F3)
        
    def forward(self, x):
        res = self.shortcut(x)
        res = self.bns(res)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        y = x + res
        y = F.relu(y)
        return y 

class resnetblock4(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(resnetblock4,self).__init__()
        F1, F2, F3 = filters
        self.conv1 = nn.Conv2d(in_channel,F1,1,stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1,F2,f,stride=1, padding=True, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(F3)
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        y = x + res
        y = F.relu(y)
        return y 

class resnet18(nn.Module):
    def __init__(self , n_channel = 1, n_output = 2):
        super(resnet18, self).__init__()
        self.conv1 = nn.Conv2d(n_channel,64,kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = resnetblock1(64)
        self.res2 = resnetblock1(64)
        self.res3 = resnetblock2(64,128)
        self.res4 = resnetblock1(128)
        self.res5 = resnetblock2(128,256)
        self.res6 = resnetblock1(256)
        self.res7 = resnetblock2(256,512)
        self.res8 = resnetblock1(512)
        self.avgp = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(512,1000)
        self.fc2 = nn.Linear(1000,n_output) 
    def forward(self,x) :
        batch_size = x.size(0)
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = F.relu(x)
        x = self.avgp(x)
        x = x.view(batch_size,-1)#flatten
        x = self.fc1(x)
        x = self.fc2(x)#(n_output)
        x = F.softmax(x,dim=1)
        # print(x)
        return x

class resnet50(nn.Module):
    def __init__(self, n_channel = 1, n_output = 2):
        super(resnet50,self).__init__()
        self.conv1 = nn.Conv2d(n_channel,64,kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block1 =nn.Sequential(
            resnetblock3(64, f=3, filters=[64, 64, 256], s=1),
            resnetblock4(256, 3, [64, 64, 256]),
            resnetblock4(256, 3, [64, 64, 256]),
        )
        self.block2 = nn.Sequential(
            resnetblock3(256, f=3, filters=[128, 128, 512], s=2),
            resnetblock4(512, 3, [128, 128, 512]),
            resnetblock4(512, 3, [128, 128, 512]),
            resnetblock4(512, 3, [128, 128, 512]),
        )
        self.block3 = nn.Sequential(
            resnetblock3(512, f=3, filters=[256, 256, 1024], s=2),
            resnetblock4(1024, 3, [256, 256, 1024]),
            resnetblock4(1024, 3, [256, 256, 1024]),
            resnetblock4(1024, 3, [256, 256, 1024]),
            resnetblock4(1024, 3, [256, 256, 1024]),
            resnetblock4(1024, 3, [256, 256, 1024]),
        )
        self.block4 = nn.Sequential(
            resnetblock3(1024, f=3, filters=[512, 512, 2048], s=2),
            resnetblock4(2048, 3, [512, 512, 2048]),
            resnetblock4(2048, 3, [512, 512, 2048]),
        )
        self.avgp = nn.AvgPool2d(7) #2,2,padding=1
        self.fc1 = nn.Linear(2048,1000)
        self.fc2 = nn.Linear(1000,n_output)
    
    def forward(self, x):
        batch_size = x.size(0)
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgp(x)
        x = x.view(batch_size,-1)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = F.softmax(x,dim=1)
        return x
    
# model = resnet50()
# from torchsummary import summary
# summary(model, input_size=[(1, 224, 224)], batch_size=128, device="cpu")