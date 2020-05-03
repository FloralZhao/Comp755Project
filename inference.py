import torch
import argparse
from PIL import Image
import time
import os

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

import GazeDetect
from train import inception
import config

import pdb



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        help='GPUs to use',
                        default='0',
                        type=str)
    args = parser.parse_args()
    return args

def accuracy(output, target, k=5):
    _, pred_k = output.topk(k, 1, True, True)
    pred_k = pred_k.t()
    correct = pred_k.eq(target.view(1, -1).expand_as(pred_k))
    correct_k = correct.view(-1).float().sum(0)
    return correct_k

def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    #### load image data
    # mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 299.
    gaze_dataset = GazeDetect.GazeDetect(type='data_random_crop_2')

    data_loader = DataLoader(gaze_dataset, batch_size=8, shuffle=True, num_workers=8, drop_last=True)


    # model = inception(config.num_classes, config.feature_extract, use_pretrained=True)
    # model.to(device)
    # model.load_state_dict(torch.load(os.path.join('./checkpoints', f'best_model_{gaze_dataset.data_type}.pth.tar')))
    # model.eval()


    print("Initialize model...")
    # model = torch.hub.load('pytorch/vision:v0.5.0', 'inception_v3', pretrained=True)
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)


    since = time.time()
    running_loss = 0.0
    running_correct_1 = 0
    running_correct_5 = 0

    for data in data_loader:
        inputs = data['image']
        labels = data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_correct_1 += torch.sum(preds == labels.data)

        _, pred_k = outputs.topk(5, 1, True, True)
        pred_k = pred_k.t()
        correct = pred_k.eq(labels.view(1, -1).expand_as(pred_k))
        correct_k = correct.view(-1).float().sum()
        running_correct_5 += correct_k
        # running_correct_5 += accuracy(outputs, labels, 5)

    time_elapsed = time.time() - since
    print('Inference time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc_1 = running_correct_1.double() / len(data_loader.dataset)
    epoch_acc_5 = running_correct_5.double() / len(data_loader.dataset)
    print('Loss: {:.4f} Top-1 Acc: {:.4f} Top-5 Acc: {:.4f}'.format(epoch_loss, epoch_acc_1, epoch_acc_5))

if __name__ == '__main__':
    main()


