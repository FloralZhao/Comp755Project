import pandas as pd
import numpy as np
import pdb

mapping_file = './classes.xlsx'
df = pd.read_excel(mapping_file, sheet_name=0)
imagenet_labels = df['imagenet_label'].tolist()
nick_labels = df['nick_label'].tolist()

imagenet_labels_text = [x.split(" ")[-1] for x in imagenet_labels]
imagenet_labels_index = [int(x.split(" ")[-2]) for x in imagenet_labels]

mapping = dict(zip(nick_labels, imagenet_labels_index))