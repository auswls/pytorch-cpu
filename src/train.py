import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import json
import argparse
import os


import pandas.util.testing as tm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


from workspace_utils import active_session


def transform_train() : 
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    return training_transforms

def transform_valid_test() : 
    transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
    return transforms


def load_dataset(file_dir, transform, flag, batch_size) : 
    dataset = datasets.ImageFolder(file_dir, transform)

    if flag == "train" : 
        data_loader = torch.utils.data.Dataloader(dataset, batch_size, shuffle=True)
    else : 
        data_loader = torch.utils.data.Dataloader(dataset, batch_size)
    
    return dataset, data_loader


def train(model, lr) : 
    # Freeze pretrained model parameters to avoid backpropogating through them
    for parameter in model.parameters():
        parameter.requires_grad = False

    from collections import OrderedDict

    # Build custom classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(5000, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return criterion, optimizer


def validation(model, validateloader, criterion):
    val_loss = 0
    accuracy = 0
    
    for images, labels in iter(validateloader):

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)
        
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy


def train_classifier(epochs, model, criterion, optimizer, train_loader, validate_loader):
    steps = 0
    print_every = 10
    model.to('cuda')

    for e in range(epochs):
        print(e)
        model.train()
        running_loss = 0

        for images, labels in iter(train_loader):
            steps += 1
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
    
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
            if steps % print_every == 0:
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validate_loader, criterion)
        
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))
        
                running_loss = 0
                model.train()


def test_accuracy(model, test_loader):
    # Do validation on the test set
    model.eval()
    model.to('cuda')

    with torch.no_grad():
        accuracy = 0
        for images, labels in iter(test_loader):
            images, labels = images.to('cuda'), labels.to('cuda')
            output = model.forward(images)
            probabilities = torch.exp(output)
            equality = (labels.data == probabilities.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print("Test Accuracy: {}".format(accuracy/len(test_loader)))


def save_checkpoint(model, training_dataset, save_path):
    model.class_to_idx = training_dataset.class_to_idx
    checkpoint = {'arch': "vgg16",
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict()
                 }
    torch.save(checkpoint, os.path.join(save_path, "checkpoint.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="path for training data")
    parser.add_argument('--save_path', type=str, help="path for saving checkpoint")
    parser.add_argument('--json_path', type=str, help="path for json file which have classes")
            
    #### Batch size ####
    parser.add_argument('--train_batch', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch', type=int, default=32, metavar='N',
            help='input batch size for testing (default: 32)')
    
    #### Epochs ####
    parser.add_argument('--epochs', type=int, default=5, metavar='N')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')


    args = parser.parse_args()

    train_dir = os.path.join(args.data_path, "train")
    test_dir = os.path.join(args.data_path, "test")
    valid_dir = os.path.join(args.data_path, "valid")

    with open(args.json_path, 'r') as f:
        flower_to_name = json.load(f)


    model = models.vgg16(pretrained=True)

    training_transforms = transform_train()
    test_transforms = transform_valid_test()
    validation_transforms = transform_valid_test()

    train_dataset, train_loader = load_dataset(train_dir, training_transforms, "train", args.train_batch)
    test_dataset, test_loader = load_dataset(test_dir, test_transforms, "test", args.test_batch)
    validation_dataset, validate_loader = load_dataset(valid_dir, validation_transforms, "valid", args.test_batch)
    

    criterion, optimizer = train(model, args.lr)

    train_classifier(args.epochs, model, criterion, optimizer, train_loader, validate_loader)
    
    test_accuracy(model, test_loader)

    save_checkpoint(model, train_dataset, args.save_path)


if __name__=='__main__' : 
    main()