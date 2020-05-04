import pandas as pd
import numpy as np
import pdb

mapping_file = './classes.xlsx'
df = pd.read_excel(mapping_file, sheet_name=0)

nick_labels = df['nick_label'].tolist()

imagenet_labels = df['imagenet_label'].tolist()
imagenet_labels_text = [x.split(" ")[-1] for x in imagenet_labels]
imagenet_labels_index = [x.split(" ")[0][1:] for x in imagenet_labels]


# mapping = dict(zip(nick_labels, imagenet_labels_index))
# # imagenet_mapping = dict(zip(imagenet_labels_index, imagenet_labels_text))
#
# with open('map_clsloc.txt') as file:
#     lines = file.readlines()
#
# class_inds = []
#
# for line in lines:
#     class_inds.append(line.split(" ")[0][1:])
# class_inds = sorted(class_inds)
#
# imagenet_mapping_sorted = dict(zip(class_inds, range(1000)))

################# another mapping
imagenet_labels_v1 = df['imagenet_label_v1'].tolist()
imagenet_labels_v1_index = [int(x.split(':')[0]) for x in imagenet_labels_v1]
mapping_v1 = dict(zip(nick_labels, imagenet_labels_v1_index))





