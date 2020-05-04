'''https://pytorch.org/hub/pytorch_vision_inception_v3/'''
import torch
import urllib
import urllib.request
import argparse
from PIL import Image
from torchvision import transforms

import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        help='GPUs to use',
                        default='0',
                        type=str)
    args = parser.parse_args()
    return args

def main():
    model = torch.hub.load('pytorch/vision:v0.5.0', 'inception_v3', pretrained=True)
    model.eval()


    #### load image data
    # mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 299.
    url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),  # range([0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    input_batch = input_batch.to(device)
    model.to(device)

    # if torch.cuda.is_available():
    #     input_batch = input_batch.to("cuda")
    #     model.to("cuda")

    with torch.no_grad():
        output = model(input_batch)
    print(output[0])
    print(torch.nn.functional.softmax(output[0], dim=0))

if __name__ == '__main__':
    main()



