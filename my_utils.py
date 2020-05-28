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
from tqdm import tqdm
import os
from config import data_transforms
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

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

'''
Remove the final layer of model
'''
def remove_last_layer(model, model_name):
    if model_name == "vgg" or model_name == "alexnet" or model_name == "squeeze":
        torch.nn.Sequential(*(list(model.model.children())[:-1]),
                            list(model.model.children())[-1][:-1])
    return torch.nn.Sequential(*(list(model.model.children())[:-1]))

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def get_feature_vector(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    csv_path = "./dataset.csv"
    test_df = pd.read_csv(csv_path)
    test_df['num_label'], _ = pd.factorize(test_df['label'])  # turn labels (string) into numbers
    test_df = test_df[test_df['type'] == "test"]
    img_path = test_df['path'].values
    labels = test_df['num_label'].values
    model, input_size = get_pretrained_models(model_name)
    model = remove_last_layer(model, model_name)
    model.eval()
    dataset_test = Alzheimer_Dataset(img_path, labels, transform=data_transforms['val'])
    dataset_loader = torch.utils.data.DataLoader(dataset_test,
                                                 batch_size=128,
                                                 shuffle=False,
                                                 num_workers=8)
    model.to(device)
    feature_vec_nps = []
    with tqdm(total=100) as pbar:
        for index, batch in enumerate(dataset_loader):
            #         print("processing: %d/%d"%(index,len(dataset_loader)))
            with torch.set_grad_enabled(False):
                inputs, labels = batch
                inputs = inputs.to(device)
                outputs = model.forward(inputs)
                for output in outputs:
                    feature_vec_nps.append(output.flatten().cpu().numpy())
            pbar.update(100 / len(dataset_loader))
    save_at = os.path.join("./feature_vectors", model_name + "_feature_vec.npy")
    print("saving feature vectors at ",save_at )
    with open(save_at, "wb") as f:
        np.save(f, feature_vec_nps)


'''
making tsne visualization with selected model
get the feature vector by forwarding testing 
images to the selected model after deleting the fully connected layer

Args: 
model_name: name of the model
sample_imgs: number of testing images to sample
    default = -1 with no sampling (using all testing images)
    should be <= total number of testing images
feature_avaiable: bool
    set to True, if "model_name"_feature_vec.npy is available in feature_vectors dir
    set to False otherwise 
'''
def tsne_visualization(model_name, sample_imgs = -1, feature_available=True):
    if not feature_available:
        get_feature_vector(model_name)
    # getting feature vectors
    feature_path = os.path.join("feature_vectors/", model_name + "_feature_vec.npy")
    with open(feature_path, "rb") as f:
        features = np.load(f)
    # load labels
    df = pd.read_csv("dataset.csv")
    df['label'], label_names = pd.factorize(df['label'])
    test_df = df[df['type'] == "test"]
    if sample_imgs != -1:
        test_df = test_df.sample(n=sample_imgs, random_state=42)
    img_path = test_df['path']
    labels = test_df['label'].values
    indices = test_df.index

    # tsne dimension reduction (2-d)
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(features[indices])

    target_ids = range(len(set(labels)))  # 4 classes

    # make plot
    plt.figure(figsize=(10, 8))
    colors = 'darkred', 'salmon', 'lightpink', 'lightblue'
    for i, c, label in zip(target_ids, colors, label_names):
        plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c=c, label=label)
    plt.title("tsne using feature vector from: " + model_name, fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()

'''
set seed to reproduce the results
optionally takes seed as argument, default seed 42
'''
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