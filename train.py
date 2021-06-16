import torch
import numpy as np
from tqdm import trange
import torch.utils.data as data 
import torch.nn as nn
import torch.optim as optim
from model import resnet18,resnet50
from dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import sys

def getAUC(y_true, y_score, task):
    if task == 'binary-class':
        y_score = y_score[:, -1]
        return roc_auc_score(y_true, y_score)

    elif task == 'multi-label, binary-class':
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        return auc / y_score.shape[1]

    else:
        auc = 0
        zero_temp = np.zeros_like(y_true)
        one_temp = np.ones_like(y_true)
        for i in range(y_score.shape[1]):
            y_true_binary = np.where(y_true == i, one_temp, zero_temp)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        return auc / y_score.shape[1]



def getACC(y_true, y_score, task, threshold=0.5):
    if task == 'multi-label, binary-class':
        zero = np.zeros_like(y_score)
        one = np.ones_like(y_score)
        y_pre = np.where(y_score < threshold, zero, one)
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        return acc / y_true.shape[1]

    elif task == 'binary-class':
        y_pre = np.zeros_like(y_true)
        for i in range(y_score.shape[0]):
            y_pre[i] = (y_score[i][-1] > threshold)
        return accuracy_score(y_true, y_pre)

    else:
        y_pre = np.zeros_like(y_true)
        for i in range(y_score.shape[0]):
            y_pre[i] = np.argmax(y_score[i])
        return accuracy_score(y_true, y_pre)

def lr_decay(optimizer, epoch, lr, rate = 0.2):
    # LR decays for every 5 epochs, with rate (default: 0.1)
    lr_new = lr * (rate ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new

def train(net, optimizer, criterion, train_loader, task, device):
    net.train()

    for idx, (input, label) in enumerate(train_loader):
        input = input.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = net(input)
        if task == 'multi-label, binary-class':
            label = label.to(torch.float32).to(device)
            loss = criterion(output, label).to(device)
        else:
            label = label.squeeze().long().to(device)
            loss = criterion(output, label).to(device)

        loss.backward()
        optimizer.step()

#测试数据
def test(net, split, data_loader, task, device):
    net.eval()
    label_true = torch.tensor([]).to(device)
    label_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for idx, (input, label) in enumerate(data_loader):
            input = input.to(device)
            label = label.to(device)
            output = net(input.to(device))

            if task == 'multi-label, binary-class':
                label = label.to(torch.float32).to(device)
                F = nn.Sigmoid()
                output = F(output).to(device)

            else:
                label = label.squeeze().long().to(device)
                F = nn.Softmax(dim = 1)
                output = F(output).to(device)
                label = label.float().resize_(len(label), 1)

            label_true = torch.cat((label_true, label), 0)
            label_score = torch.cat((label_score, output), 0)
            
    label_true = label_true.cpu().numpy()
    label_score = label_score.detach().cpu().numpy()

    auc = getAUC(label_true, label_score, task)
    acc = getACC(label_true, label_score, task)
    print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))


#验证数据
def val(net, val_loader, val_auc_list, task, device):
    net.eval()
    label_true = torch.tensor([]).to(device)
    label_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for idx, (input, label) in enumerate(val_loader):
            input = input.to(device)
            label = label.to(device)
            output = net(input.to(device))

            if task == 'multi-label, binary-class':
                label = label.to(torch.float32).to(device)
                F = nn.Sigmoid()
                output = F(output).to(device)

            else:
                label = label.squeeze().long().to(device)
                F = nn.Softmax(dim = 1)
                output = F(output).to(device)
                label = label.float().resize_(len(label), 1)

            label_true = torch.cat((label_true, label), 0)
            label_score = torch.cat((label_score, output), 0)


    label_true = label_true.cpu().numpy()
    label_score = label_score.detach().cpu().numpy()
    auc = getAUC(label_true, label_score, task)
    acc = getACC(label_true, label_score, task)
    val_auc_list.append(auc)
    # print(' AUC: %.5f ACC: %.5f' %  (auc, acc))


#提取数据
if __name__ == "__main__":
    print("Please input which mnist do you want to train:\n")
    print(" pathmnist:1\n")
    print(" chestmnist:2\n")
    print(" dermamnist:3\n")
    print(" octmnist:4\n")
    print(" pneumoniamnist:5\n")
    print(" retinamnist:6\n")
    print(" breastmnist:7\n")
    print(" organmnist_axial:8\n")
    print(" organmnist_coronal:9\n")
    print(" organmnist_sagittal:10\n")
    flag = input("please input:")
    flagclass = {
            "1": PathMNIST,
            "2": ChestMNIST,
            "3": DermaMNIST,
            "4": OCTMNIST,
            "5": PneumoniaMNIST,
            "6": RetinaMNIST,
            "7": BreastMNIST,
            "8": OrganMNISTAxial,
            "9": OrganMNISTCoronal,
            "10": OrganMNISTSagittal,
            }
    task_types = {
            "1": "multi-class",
            "2": "multi-label, binary-class",
            "3": "multi-class",
            "4": "multi-class",
            "5": "binary-class",
            "6": "ordinal regression",
            "7": "binary-class",
            "8": "multi-class",
            "9": "multi-class",
            "10": "multi-class",
            }
    infoclass = {
            "1": "pathmnist",
            "2": "chestmnist",
            "3": "dermamnist",
            "4": "octmnist",
            "5": "pneumoniamnist",
            "6": "retinamnist",
            "7": "breastmnist",
            "8": "organmnist_axial",
            "9": "organmnist_coronal",
            "10": "organmnist_sagittal",
            }
    channels = {
            "1": 3,
            "2": 1,
            "3": 3,
            "4": 1,
            "5": 1,
            "6": 3,
            "7": 1,
            "8": 1,
            "9": 1,
            "10": 1,
            }
    outputs = {
            "1": 9,
            "2": 14,
            "3": 7,
            "4": 4,
            "5": 2,
            "6": 5,
            "7": 2,
            "8": 11,
            "9": 11,
            "10": 11,
            }
    task = task_types[flag]
    n_channel = channels[flag]
    n_output = outputs[flag]
    print("Please input which model do you want to use:\n")
    print("resnet18:1\n")
    print("resnet50:2\n")
    modelflag = input("please input:")

    #resize data
    train_transform = transforms.Compose([
             # transforms.Resize([224,224]),
             transforms.RandomResizedCrop(224),
             transforms.RandomRotation(degrees=15),
             # transforms.RandomHorizontalFlip(),
             # transforms.RandomVerticalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])
             ])

    val_transform = transforms.Compose(
            [transforms.Resize([224,224]),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])
             ])
    
    test_transform = transforms.Compose(
            [transforms.Resize([224,224]),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])
             ])
    
    DataClass = flagclass[flag]
    input_root = "./npz"
    if modelflag == '1':
        batch_size = 128
    elif modelflag == '2':
        batch_size = 32

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #这里进行数据读取和数据处理，比如transform 
    print('==> Preparing the %d data, model flag is %d...'%(int(flag), int(modelflag)))
    train_dataset = DataClass(root=input_root,#文件路径
                                        split='train',
                                        transform=train_transform,
                              )
    
    train_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
    
    val_dataset = DataClass(root=input_root,
                                      split='val',
                                      transform=val_transform)
    
    val_loader = data.DataLoader(dataset=val_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
    
    test_dataset = DataClass(root=input_root,
                                        split='test',
                                        transform=test_transform)
    
    test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
    
    print('==> Building and training model...')
    #训练数据
    lr= 0.001

    if (modelflag == '1'):
        model = resnet18(n_channel=n_channel, n_output=n_output).to(device)
    elif (modelflag == '2'):
        model = resnet50(n_channel=n_channel, n_output=n_output).to(device)
            
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
                    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    epochs = 40
    val_auc_list = []
    print('==> Training model...')
    for epoch in trange(start_epoch, epochs):
        lr_decay(optimizer, epoch, lr)
        train(model, optimizer, criterion, train_loader, task, device)
        val(model, val_loader, val_auc_list, task, device)
                        
    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))
                        
    print('==> Testing model...')
    test(model,'train', train_loader, task, device)
    test(model, 'val', val_loader, task, device)
    test(model,'test', test_loader, task, device)
