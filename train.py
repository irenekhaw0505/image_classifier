# Imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from PIL import Image
import json
import argparse



parser = argparse.ArgumentParser()

parser.add_argument('--data_directory', help = 'specify data directory folder', default='./flowers')
parser.add_argument('--arch', type = str, help = 'set CNN architecture', default='vgg13')
parser.add_argument('--learning_rate', type=float, help = 'set model learning rate', default=0.001)
parser.add_argument('--hidden_input', type=int, help = 'set hidden units', default=512)
parser.add_argument('--epochs', type=int, help = 'set model epoch count', default=20)
parser.add_argument('--save_dir', help = 'set model checkpoint name', default='checkpoint.pth')
parser.add_argument('--gpu', help = 'set gpu to use gpu computation', default="gpu")

args = parser.parse_args()

data_dir = args.data_directory
nn_arch = args.arch
hidden_units = args.hidden_input
learning_rate = args.learning_rate
epoch_num = args.epochs
training_mode = args.gpu


def transform_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_trans = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_trans = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_trans = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) 

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_trans)
    test_datasets  = datasets.ImageFolder(test_dir,  transform=test_trans)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_trans)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_datasets,  batch_size=64)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    
    return train_loader, valid_loader,train_datasets
    
def get_category_labels():
    with open('cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)
        print(cat_to_name)
        
    return cat_to_name
        
        
def get_model(nn_arch):
    
    if nn_arch == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} is not a correct architecture. Please choose either vgg13 or alexnet.".format(nn_arch))
        
    for param in model.parameters():
        param.requires_grad = False
        
    return model
        
        
def validation_check(criterion, model, training_mode):
    correct, total, loss = 0, 0, 0

    with torch.no_grad():
        for images, labels in valid_loader:
            
            if torch.cuda.is_available():
                if training_mode == 'gpu':
                    images = images.to('cuda')
                    labels = labels.to('cuda')
                       
            output = model.forward(images)
            loss += criterion(output, labels).item()
            probability = torch.exp(output) 
            prediction = probability.max(dim = 1) 

            matches = (prediction[1] == labels.data)
            correct += matches.sum().item()
            total += 64
            accuracy = 100 * (correct / total)
        
        return loss, accuracy
    
    
    
def train_model(model, train_data_loader, valid_data_loader, optimizer, criterion, epochs, training_mode):
    
    epoch_count = 0
    steps = 0
    print_every = 50
    valid_size = len(valid_data_loader)

    while epoch_count != epochs:
        running_loss = 0
        validation_loss = 0
        for inputs, labels in iter(train_data_loader):
            steps += 1
            
            if torch.cuda.is_available():
                if training_mode == 'gpu':
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
           
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                validation_loss, accuracy = validation_check(criterion, model, training_mode)
                
                print("Epoch= {} of {},".format(epoch_count + 1, epochs), 
                      "Loss= {:.3f},".format(running_loss / print_every),
                      "Validation Set Loss= {:.3f},".format(validation_loss / valid_size),
                      "Validation Set Accuracy= {:.3f}".format(accuracy))
                running_loss = 0
                model.train()
                
        epoch_count += 1
 
        
def create_nn_model(model, cat_to_name, train_loader, valid_loader, train_datasets, learning_rate, hidden_unit, training_mode, epoch_num):
    input_size = model.classifier[0].in_features 
    output_size = len(cat_to_name)
    hidden_layer1 = 1600
    hidden_layer2 = hidden_unit
    epoch_total = epoch_num
    drop = 0.2
    print_every = 50

    # create model classifier
    classifier = nn.Sequential(nn.Linear(input_size, hidden_layer1),
                               nn.ReLU(),
                               nn.Dropout(p = 0.15),
                               nn.Linear(hidden_layer1, hidden_layer2),
                               nn.Dropout(p = 0.15),
                               nn.Linear(hidden_layer2, output_size),
                               nn.LogSoftmax(dim = 1))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    if torch.cuda.is_available():
        if training_mode == 'gpu': 
           model.cuda()
     
    train_model(model, train_loader, valid_loader, optimizer, criterion, epoch_total, training_mode)

    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layer1': hidden_layer1,
                  'hidden_layer2': hidden_layer2,
                  'epoch_total': epoch_total,
                  'drop': drop,
                  'classifier': classifier,
                  'state_dict': model.state_dict(),
                  'optimizer_dict': optimizer.state_dict(),
                  'class_to_idx': train_datasets.class_to_idx,
                  'arch': nn_arch,
                  'learning_rate': learning_rate,
                 }

    torch.save(checkpoint, 'checkpoint.pth')
    

    
   
train_loader, valid_loader, train_datasets = transform_data(data_dir)
cat_to_name = get_category_labels()
model = get_model(nn_arch)
create_nn_model(model, cat_to_name,train_loader, valid_loader, train_datasets, learning_rate, hidden_units, training_mode, epoch_num)

