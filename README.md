# Comp755Project
The final project for COMP755. 

## Prerequisites
1. python >= 3.6.9
2. Install dependencies
```
pip install -r requirements.txt
```

## Usage
1. Finetune pretrained inception-v3 using our data.
```
python train.py --gpu 0
```

2. Inference using model pretrained on ImageNet or finetuned on our data.
```
python inference.py --gpu 0
```
