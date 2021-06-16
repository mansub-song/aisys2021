'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-m', default="DPN92", type=str, help='model name')
parser.add_argument('-p', type=str, help='file path')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
file_path = args.p
model_name = args.m
storage_print = {"/mnt/pmem0":"PMEM", "/mnt/sda":"SATA", ".":"NVME"}
model_print = {"DPN92":" DPN92 ", "VGG":"  VGG  ", "S_DLA":" S_DLA_"}
print("-"*13+model_print[model_name]+storage_print[file_path]+" "+"-"*14)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
#print('==> Preparing data..')
print('|          Preparing data..           |')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

startTime = time.time()
trainset = torchvision.datasets.CIFAR10(
    root=file_path+'/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
duTime = time.time() - startTime
dataSave = pd.read_csv("/home/aisys/checkpoint.csv").drop(['Unnamed: 0'], axis = 1)
dnV = {'time':[duTime]}
dataSave2 = pd.DataFrame(data=dnV)
dataSave = pd.concat([dataSave, dataSave2], axis = 0)


startTime = time.time()

testset = torchvision.datasets.CIFAR10(
    root=file_path+'/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
duTime = time.time() - startTime
print("|    TestData load Time:{:.8f}    |".format(duTime))
dnV = {'time':[duTime]}
dataSave2 = pd.DataFrame(data=dnV)
dataSave = pd.concat([dataSave, dataSave2], axis = 0)
dataSave.to_csv("/home/aisys/checkpoint.csv")



classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
#print('==> Building model..')
model_dict={"DPN92":DPN92(), "VGG":VGG('VGG19'), "S_DLA":SimpleDLA()}
#net = VGG('VGG19')
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
#net = MobileNetV2()
#net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
#model_name = "simpleDLA"
net = model_dict[model_name]
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#if args.resume:
if args.resume:
    # Load checkpoint.
    # Load checkpoint.
    print('|  Resuming from checkpoint at *{}  |'.format(storage_print[file_path]))
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('{}/checkpoint/{}_ckpt.pth'.format(file_path, model_name),map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    #if acc > best_acc:
    if True:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.join('checkpoint',model_name)):
            os.mkdir(os.path.join('checkpoint',model_name))
        #torch.save(state, './checkpoint/{}/ckpt_{}.pth'.format(model_name,epoch))
        torch.save(state, '{}/checkpoint/{}_ckpt.pth'.format(file_path, model_name))
        best_acc = acc


#for epoch in range(start_epoch, start_epoch+1):
    #train(epoch)
    #test(epoch)
    #scheduler.step()
