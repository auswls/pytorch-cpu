import numpy as np
import pandas as pd
import cv2
from PIL import Image
import json

import pandas.util.testing as tm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


from collections import OrderedDict
import argparse


# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location="cpu")
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Architecture not recognized.")
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(5000, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)
    
    # Resize
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))
        
    # Crop 
    left_margin = (pil_image.width-224)/2
    bottom_margin = (pil_image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image



### Class Prediction ###

# Implement the code to predict the class from an image file

def predict(image_path, model, topk=5, gpu="cpu"):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(gpu)
    
    image = process_image(image_path)
    
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.FloatTensor)
    
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    output = model.forward(image.to(gpu))
    probabilities = torch.exp(output)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print(idx_to_class)
    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes
    



def main():
    print("this is main test code")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type = str, default = 'images/flowers/test/15/image_06351.jpg', help="path for input data to predict")
    parser.add_argument('--save_path', type=str, default = 'weights/checkpoint.pth', help="path for saving checkpoint")
    parser.add_argument('--json_path', type = str, default = 'flower_to_name.json', help="path for json file which have classes")

    parser.add_argument('--gpu', default="cpu", help='disables CUDA training')


    args = parser.parse_args()


    device = torch.device("cpu")

    with open(args.json_path, 'r') as f:
        flower_to_name = json.load(f)


    model = load_checkpoint(args.save_path)
    probs, classes = predict(args.input_path, model.to(device), 5, args.gpu)   

    """
    for i in range(len(classes)) : 
        print(classes[i], " : ", flower_to_name[classes[i]])
    """

    print("flower : ", flower_to_name[classes[0]], " / probability : ", probs[0])

    if probs[0] > 0.8 : 
        result = open("prediction_result/result.txt", "w")
        data = "{} : {}" .format(flower_to_name[classes[0]], probs[0])
        result.write(data)
    else : 
        print("probability is too low")

 
if __name__=='__main__' : 
    main()