from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import pandas as pd
from my_utils import *
from config import MODELS, SAVE_HIST, SAVE_MODELS, SAVE_OPT, MAX_EPs, BATCH_SIZE, data_transforms
# skorch
import skorch
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint
from sklearn.model_selection import train_test_split

# Ignore warnings

import warnings
warnings.filterwarnings("ignore")
import os


def setup_dirs():
    for dir in ["models", "optimizers", "outputs", "train_histories", "feature_vectors"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

if __name__ == "__main__":
    setup_dirs()
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    lrscheduler = LRScheduler(policy='StepLR', step_size=10, gamma=0.1)
    early_stopping = EarlyStopping(monitor="valid_loss", patience=5, threshold=0.01)
    csv_path = "./dataset.csv"
    train_df = pd.read_csv(csv_path)
    train_df['num_label'], _ = pd.factorize(train_df['label'])  # turn labels (string) into numbers
    train_df = train_df[train_df['type'] == "train"]
    img_path = train_df['path'].values
    labels = train_df['num_label'].values
    X_train, X_val, y_train, y_val = train_test_split(img_path, labels, test_size=0.1,
                                                      stratify=labels, random_state=42)
    print("train set: %d, test set : %d" %(len(X_train), len(X_val)))
    for model, batch_size, hist, save_model, save_opt, ep in zip(MODELS, BATCH_SIZE, SAVE_HIST,
                                                                SAVE_MODELS, SAVE_OPT, MAX_EPs):
        if model == "alexnet": # Alexnet requires lower lr
            lr = 0.001
        else:
            lr = 0.01
        print("Training with model ", model)
        model, input_size = get_pretrained_models(model)
        dataset_train = Alzheimer_Dataset(X_train, y_train, transform=data_transforms['train'])
        dataset_val = Alzheimer_Dataset(X_val, y_val, transform=data_transforms['train'])

        checkpoint = Checkpoint(f_params=save_model,
                                monitor='valid_acc_best',
                                f_optimizer=save_opt,
                                f_history=hist)
        seed_everything = FixRandomSeed()
        net = NeuralNetClassifier(model,
                                criterion=nn.CrossEntropyLoss,
                                optimizer=optim.SGD,
                                lr=lr,
                                batch_size=batch_size,
                                max_epochs=ep,
                                optimizer__momentum=0.90,
                                iterator_train__shuffle=True,
                                iterator_train__num_workers=8,
                                iterator_valid__shuffle=True,
                                iterator_valid__num_workers=8,
                                train_split=predefined_split(dataset_val),
                                callbacks=[lrscheduler, checkpoint, seed_everything, early_stopping],
                                device=device)
        # split once only
        net.fit(dataset_train, y=None)