# Comp755Project
The final project for COMP755. 

## Prerequisites
1. python >= 3.6.9
2. Install dependencies
```
pip install -r requirements.txt
```
3. Download our synthetic dataset [here](https://github.com/nrewkowski/COMP755FinalProject)

## Usage
1. Finetune pretrained inception-v3 using our data.

Finetuning the pretrained model
```
python train_finetune.py --gpu 0
```
Inference using finetuned model
```
python inference_finetune.py
```

2. Inference using model pretrained on ImageNet.
```
python inference.py --gpu 0
```
