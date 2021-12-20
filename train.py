import argparse

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder

import torch.nn.functional as F

from PIL import Image
from collections import OrderedDict

import time
import numpy as np
import matplotlib.pyplot as plt

from util import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, epochs, gpu):         # dataloaders[0] = train, dataloaders[1] = validation, dataloaders[2] = test
    steps = 0
    print_every = 10
    for e in range(epochs):
        running_loss = 0        
        for ii, (inputs, labels) in enumerate(dataloaders[0]):
            steps += 1 
            
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda')     # Move input and label tensors to the default device(use cuda)
            else:
                model.cpu()                                               # Use cpu other than 'gpu'
            
            # Zeros the gradients on each training pass
            optimizer.zero_grad()     
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            # Use the logits to calculate the loss
            loss = criterion(outputs, labels)
            # Perform a backward pass through the network to calculate the gradients
            loss.backward()
            # Take a step with the optimizer to update the weights
            optimizer.step()
            
            # Calculate the training loss
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valloss = 0
                accuracy= 0

                for ii, (inputs2,labels2) in enumerate(dataloaders[1]):
                        optimizer.zero_grad()
                        
                        if gpu == 'gpu':
                            inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') # Use cuda
                            model.to('cuda:0') 
                        else:
                            # Use the inputs
                            pass 
                            
                        with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            valloss = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                valloss = valloss / len(dataloaders[1])
                accuracy = accuracy /len(dataloaders[1])

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Train Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(valloss),
                      "Test accuracy: {:.4f}".format(accuracy),
                     )

                running_loss = 0
            
def main():
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
    
    validataion_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])]) 

    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])]) 

    image_datasets = [ImageFolder(train_dir, transform=train_transforms),
                      ImageFolder(val_dir, transform=validataion_transforms),
                      ImageFolder(test_dir, transform=test_transforms)]
    
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg16":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, 4096)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(0.2)),
                                  ('fc2', nn.Linear(4096, 512)),
                                  ('relu', nn.ReLU()),
                                  ('fc3', nn.Linear(512, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('dropout', nn.Dropout(p=0.6)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    # Update the classifier in the model    
    model.classifier = classifier
    # Define the loss
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = image_datasets[0].class_to_idx
    # Get the gpu settings
    gpu = args.gpu 
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    # New save location
    path = args.save_dir  
    save_checkpoint(path, model, optimizer, args, classifier)


if __name__ == "__main__":
    main()