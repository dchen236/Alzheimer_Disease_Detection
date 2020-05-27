from __future__ import print_function, division
import pandas as pd
from skimage import io, transform
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets,models
import torch.nn as nn
from skorch.callbacks import Callback
import random
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
'''
Customized Dataset:
input: 
    dataset_csv: csv file with col: path (image path) and col: label 
    transform: torch transform, None by default
    type: set type to train during training, set to test during testing
    img_path: extracted from dataset.csv using df['path']
    img_path: extracted from dataset.csv using df['label']
'''
class Alzheimer_Dataset(Dataset):
    def __init__(self, img_path, labels, transform=None):
        self.transform = transform
        self.imgs = img_path
        self.labels = labels
        self.length = len(self.imgs)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        img_path = self.imgs[index]
        image = io.imread(img_path)
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
'''
    initialize models
    Adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
'''
def get_pretrained_models(model_name):
    NUM_OUTPUT = 4
    class Pretrained_Model(nn.Module):
        def __init__(self,  output_features, model_name):
            super().__init__()
            if model_name == "res18":
                model = models.resnet18(pretrained=True)
            elif model_name == "res34":
                model = models.resnet34(pretrained=True)
            elif model_name == "res50":
                model = models.resnet50(pretrained=True)
            elif model_name == "res101":
                model = models.resnet101(pretrained=True)
            elif model_name == "res152":
                model = models.resnet152(pretrained=True)
            elif model_name == "squeeze":
                model = models.squeezenet1_0(pretrained=True)
            elif model_name == "vgg":
                model = models.vgg11_bn(pretrained=True)
            elif model_name == "alexnet":
                model = models.alexnet(pretrained=True)
            elif model_name == "densenet":
                model = models.densenet121(pretrained=True)
            else:
                print("wrong name, quiting")
                exit()
            # adjusting model archetectures
            if model_name.startswith("res") or model_name == "densenet":
                # pretrained models take rgb image, we have gray image, thus 1, 64 instead of 3, 64
                if model_name == "densenet":
                    model.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, output_features)
                else: # resnets
                    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, output_features)
            elif model_name == "squeeze":
                model.features[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2)
                model.classifier[1] = nn.Conv2d(512, output_features, kernel_size=(1, 1), stride=(1, 1))
                model.num_classes = output_features
            elif model_name == "vgg" or model_name == "alexnet":
                if model_name == "alexnet":
                    model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
                else:
                    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs,output_features)
            self.model = model
        def forward(self, x):
            return self.model(x)
    input_size = 244
    return Pretrained_Model(NUM_OUTPUT, model_name), input_size

class FixRandomSeed(Callback):
    def __init__(self, seed=42):
        self.seed = seed
    def initialize(self):
        # print("setting random seed to: ", self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True