from __future__ import print_function, division
import os

from PIL import Image
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from mapping import nick_labels, imagenet_labels_text, imagenet_labels_index, mapping

import pdb


data_types = ['data_random_crop_2',
                 'data_random_crop_3',
                 'data_random_foveat',
                 'data_random_gaussian_2',
                 'data_random_gaussian_3',
                 'data_random_original',
                 'data_spline_original']

# classes = ['wall', 'door', 'book', 'mug', 'picture frame', 'sliding doors', 'curtain',
#            'shelves', 'plant', 'plate', 'vase', 'floor lamp', 'wooden floor', 'tv',
#            'couch', 'coat hook', 'coffee table', 'light switch', 'statue', 'power outlet',
#            'floor moulding', 'desk lamp', 'ceiling light', 'coat hook backing', 'carpet']

class GazeDetect(Dataset):
    def __init__(self, type, root_dir='/playpen/dongxuz1/gaze_object_detection/dataset', transform=None):

        assert type in data_types, f"Invalid data type {type}."
        self.data_type = type

        self.root_dir = os.path.join(root_dir, type)
        self.class_map = mapping

        if transform is None:
            if self.data_type.endswith('original'):
                self.transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        image_list = glob.glob(os.path.join(self.root_dir, '*.png'))
        self.images = []
        for image in image_list:
            img_name = os.path.basename(image)
            label_name = img_name.split('_')[1][1:-1]
            if label_name in nick_labels:
                self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')
        img_name = os.path.basename(img_name)
        label_name = img_name.split('_')[1][1:-1]
        label = self.class_map[label_name]

        if self.transform:
            image = self.transform(image)


        sample = {'img_name': img_name, 'image': image, 'label': label}


        return sample

if __name__ == '__main__':
    gaze_dataset = GazeDetect(type='data_random_original')
    dataloader = DataLoader(gaze_dataset, batch_size=8, shuffle=True, num_workers=8)
    data = gaze_dataset[6]
    pdb.set_trace()



