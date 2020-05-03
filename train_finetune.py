from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn

import time
import copy
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse

import GazeDetect_finetune
import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        default='0',
                        type=str)
    args = parser.parse_args()
    return args

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, is_inception=True):
    '''

    :param model:
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :param is_inception: a flag used to accomodate the Inception v3 model, as that architecture uses an auxiliary output and the overall model loss respects both the auxiliary output and the final output
    :return:
    '''

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Interate over data
            for data in dataloaders[phase]:
                inputs = data['image']
                labels = data['label']
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # get model outputs and calculate loss
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)  # return (values, indices)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                scheduler.step(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc:{:.4f}'.format(best_acc))

    # load best model weight
    model.load_state_dict(best_model_wts)

    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    # By default, when we load a pretrained model all of the parameters have .requires_grad=True
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'inception':
        '''
        expects (299, 299) sized images and has auxiliary output
        '''
        print("Initialize model...")
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxiliary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print.info("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def inception(num_classes, feature_extract, use_pretrained=True):
    '''
    expects (299, 299) sized images and has auxiliary output
    '''
    print("Initialize model...")
    model_ft = models.inception_v3(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    # Handle the auxiliary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 299



    return model_ft


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        # device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    cudnn.benckmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    cpt_dir = './checkpoints'
    if not os.path.exists(cpt_dir):
        os.mkdir(cpt_dir)


    model_ft = inception(config.num_classes, config.feature_extract, use_pretrained=True)
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    if config.feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)



    print("Initializing Datasets and Dataloaders...")
    gaze_dtaset = GazeDetect_finetune.GazeDetect(type='data_random_gaussian')
    trainset, valset = random_split(gaze_dtaset, [4000, 817])
    train_dataloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)
    # val_dataset = GazeDetect.GazeDetect(type='data_random_crop_3')
    val_dataloader = DataLoader(valset, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}


    optimizer_ft = optim.SGD(params_to_update, lr=config.lr, momentum=config.momentum)
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, threshold=0.001,
                                                     factor=0.5)
    criterion = nn.CrossEntropyLoss()

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler_ft, device, num_epochs=config.num_epochs, is_inception=(config.model_name=="inception"))
    torch.save(model_ft.state_dict(), f'./{cpt_dir}/best_model_{gaze_dtaset.data_type}.pth.tar')

    ohist = [h.cpu().numpy() for h in hist]
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, config.num_epochs + 1), ohist)
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, config.num_epochs + 1, 1.0))
    plt.savefig(f'./fig_{gaze_dtaset.data_type}.png')

if __name__ == '__main__':
    main()