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
import cat_label
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--image_path', help = 'specify image path', default='./flowers/valid/95/image_07585.jpg')
parser.add_argument('--top_k', help = 'set top k value', type=int, default=5)
parser.add_argument('--checkpoint', help = 'choose nn model checkpoint name', default='checkpoint.pth')
parser.add_argument('--gpu', help = 'set gpu to use gpu computation', default="gpu")

args = parser.parse_args()

image_path = args.image_path
nn_checkpoint = args.checkpoint
top_k = args.top_k


def load_and_rebuild_model(nn_checkpoint):
    
    checkpoint = torch.load(nn_checkpoint)
    arch = checkpoint['arch']
        
    if arch == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} is not a correct architecture. Please choose either vgg13 or alexnet.".format(arch))
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    processed_img = Image.open(image)  
    preprocess_img = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                        ])
    
    processed_img = preprocess_img(processed_img)    
    return processed_img



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()   
    model.cpu() 
    prediction_image = process_image(image_path).unsqueeze(0)  
    
    with torch.no_grad():
        output = model.forward(prediction_image)
        probability, labels = torch.topk(output, topk)        
        probability = probability.exp()
        
        class_idx = {model.class_to_idx[k]: k for k in model.class_to_idx}
        
        probable_classes = []
        
        for label in labels.numpy()[0]:
            probable_classes.append(class_idx[label])

        return probability.numpy()[0], probable_classes
    
def print_result(probability, probable_classes, top_k=5):
    cat_to_name = cat_label.load_json_data()
    flower_classes = [cat_to_name[name] for name in probable_classes]    
    print('\n---------- Top 5 Classes Predicted on the image ----------\n')
    print('Highest Probability Predicted Flower Class: {}\n'.format(flower_classes[0]))
    for i in range(top_k):
        print("{:.6f}% of {}".format(probability[i] * 100, [cat_to_name[name] for name in probable_classes][i]))
    print('\n')
    
existing_model = load_and_rebuild_model(nn_checkpoint)
probability, probable_classes = predict(image_path, existing_model, top_k)
print_result(probability, probable_classes, top_k)

