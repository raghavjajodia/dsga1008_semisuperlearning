#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import random
import numpy as np
plt.ion()   # interactive mode



#General Code for supervised train
def train_model(model, criterion, optimizer, scheduler, device, checkpoint_path, f, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        f.write('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
        f.write('-' * 10)
        f.write('\n')
        f.flush()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_correct_top5 = 0
            n_samples = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                batchSize = inputs.size(0)
                n_samples += batchSize

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                pred_top_1 = torch.topk(outputs, k=1, dim=1)[1]
                running_corrects += pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()
                pred_top_k = torch.topk(outputs, k=top_k, dim=1)[1]
                target_top_k = labels.view(-1, 1).expand(batchSize, top_k)
                running_correct_top5 += pred_top_k.eq(target_top_k).int().sum().item()
                
            # Metrics
            top_1_acc = running_corrects/n_samples
            top_k_acc = running_correct_top5/n_samples
            epoch_loss = running_loss / n_samples

            f.write('{} Loss: {:.4f} Top 1 Acc: {:.4f} Top k Acc: {:.4f}\n'.format(phase, epoch_loss, top_1_acc, top_k_acc))
            f.flush()

            # deep copy the model
            if phase == 'val' and top_k_acc > best_acc:
                best_acc = top_k_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), '%s/net_epoch_%d.pth' % (checkpoint_path, epoch))

    time_elapsed = time.time() - since
    f.write('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    f.write('Best val Acc: {:4f} \n'.format(best_acc))
    f.flush()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='', help="path to net (to initialize)")
parser.add_argument('--netCont', default='', help="path to net (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
f = open("{}/training_logs.txt".format(opt.outf),"w+")

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
f.write("Random Seed: {} \n".format(opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


cudnn.benchmark = True
ngpu = int(opt.ngpu)
top_k = 5


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(opt.imageSize,scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(opt.dataroot, x), data_transforms[x]) for x in ['train', 'val']}

assert image_datasets

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= opt.batchSize, pin_memory= True, shuffle=True, num_workers=opt.workers) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

if torch.cuda.is_available() and not opt.cuda:
    f.write("WARNING: You have a CUDA device, so you should probably run with --cuda \n")

f.flush()

device = torch.device("cuda:0" if opt.cuda else "cpu")


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Discriminator(nn.Module):
    def __init__(self, ngpu, num_class):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flattened = Flatten()
        self.dropout = nn.Dropout(p=0.6)
        self.linear = nn.Linear(opt.ndf * 8 * 4 * 4, num_class)            

    def forward(self, input):
        if input.cuda and self.ngpu > 1:
            output1 = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output2 = nn.parallel.data_parallel(self.flattened, output1, range(self.ngpu))
            output3 = nn.parallel.data_parallel(self.dropout, output2, range(self.ngpu))
            output4 = nn.parallel.data_parallel(self.linear, output3, range(self.ngpu))
        else:
            output1 = self.main(input)
            output2 = self.flattened(output1)
            output3 = self.dropout(output2)
            output4 = self.linear(output3)
            
        return output4

netD = Discriminator(opt.ngpu, len(class_names)).to(device)

if opt.netCont !='':
    netD.load_state_dict(torch.load(opt.netCont, map_location=device))
    f.write('Loaded state and continuing training')
elif opt.net !='':
    netD.load_state_dict(torch.load(opt.net, map_location=device), strict=False)
    f.write('initialized state with pretrained net')

for param in netD.parameters():
    param.requires_grad = False
    
for param in netD.linear.parameters():
    param.requires_grad = True


criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.Adam(netD.linear.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
netD = train_model(netD, criterion, optimizer_conv, exp_lr_scheduler, device, opt.outf, f, num_epochs=opt.niter)
torch.save(netD.state_dict(), '%s/netD_best_weights.pth' % (opt.outf))
f.close()