import numpy as np
import itertools
import json
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
                            f1_score, precision_score, recall_score, \
                            confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from config import MODELS
sns.set(style="dark")

"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(cm,
                          m_ax,
                          classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          ):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.set(font_scale=1.2) # for label size
    subfig = sns.heatmap(cm, annot=True,
                xticklabels = classes,
                yticklabels = classes,
                annot_kws={"size": 15},
                cmap = cmap, ax = m_ax) # font size
    subfig.set_yticklabels(subfig.get_yticklabels(), rotation=90)

'''
This function reports the performance of model based on plain and weighted arrucacy
precision, racall and f1-score
see: https://scikit-learn.org/stable/modules/model_evaluation.html for metrics explanations
Args: 
    target: label (np.array or list)
    preds: model output / prediction (np.array or list)
    binary: True or False (default)
        When binary = True, turns labels into binary (Demented or Not Demented)
        labels has to be binary when binairy = True
'''
def metrics_overall(target, preds, binary = False):
    if binary:
        assert(set(target) == {0, 1} and set(preds) == {0, 1})
        average = "binary"
    else:
        average = "weighted"
    acc = accuracy_score(target, preds)
    binary_acc = balanced_accuracy_score(target, preds)
    pre = precision_score(target, preds, average=average)
    recall = recall_score(target, preds, average=average)
    f1 = f1_score(target, preds, average=average)
    return acc, binary_acc, pre, recall, f1


'''
Report performance in temrs of metrics and confusion matrix
This will report multiclass performance and binary performance (Demented or Not Demented)
Args:
    title: name of the testing model
    target: ground truth
    preds: prediction from model
'''
def report_performance(binary = False):
    model_performances = {}
    for model in MODELS:
        output_df = pd.read_csv("outputs/" + model + "_outputs.csv")
        target, preds = output_df['label'], output_df['output']
        if binary: # turn into demented and not demented (1 and 0)
            target  = [0 if label == 3 else 1 for label in target]  # 0 not demented
            preds = [0 if label == 3 else 1 for label in preds]  # 0 not demented
        acc, binary_acc, pre, recall, f1 = metrics_overall(target, preds, binary)
        model_performances[model] = {}
        model_performances[model]['accuracy'] = round(acc, 2)
        model_performances[model]['binary_acc'] = round(binary_acc, 2)
        model_performances[model]['precision'] = round(pre, 2)
        model_performances[model]['recall'] = round(recall, 2)
        model_performances[model]['f1'] = round(f1, 2)
    return pd.DataFrame(model_performances)

'''
Plot two confusion matrices side by side, 
left plot: 4 classes (VeryMild, Moderate, Mild, NonDemented)
right plot: 2 classes (Demented, NonDemented)
Args:
    title: model name
    target: labels
    preds: model predictions
'''
def side_by_side_confusion_matrix(title, target, preds, figsize=(15, 10)):
    labels = ['VeryMild', 'Moderate', 'Mild', 'NonDemented']
    labels_bi = ["NonDemented", "Demented"]
    target_bi = [0 if label == 3 else 1 for label in target]  # 0 not demented
    preds_bi = [0 if label == 3 else 1 for label in preds]  # 0 not demented
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plot_confusion_matrix(confusion_matrix(target, preds), m_ax=ax1, classes=labels)
    plot_confusion_matrix(confusion_matrix(target_bi, preds_bi), m_ax=ax2, classes=labels_bi)
    plt.tight_layout(pad=3)
    fig.suptitle(title, fontsize = 15)
    plt.show()

'''
extract train loss, validation loss and validation acc 
from json files in train_history folder
Args:
    model_name: name of the model
'''
def extract_train_history(model_name):
    json_path = "train_histories/" + model_name + "_hist.json"
    with open(json_path, "r") as j:
        hist = json.load(j)
    train_hist = {"train_loss": [],
                  "valid_loss": [],
                  "valid_acc": []}
    for ep in hist:
        train_hist['train_loss'].append(ep['train_loss'])
        train_hist['valid_loss'].append(ep['valid_loss'])
        train_hist['valid_acc'].append(ep['valid_acc'])
    return train_hist

'''
Plot train loss, validition loss and validation arruracy side by side
Args:
    train_hist: dictionary extracted from extract function
    title: name of the model
'''
def plot_train_hist(train_hist, title):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax1.plot(train_hist['train_loss'], label="train")
    ax1.plot(train_hist['valid_loss'], label="val_loss")
    ax1.set_title("loss")
    ax1.legend()
    ax2.plot(train_hist['valid_acc'], label="valid_acc")
    ax2.legend()
    ax2.set_title("accuracy")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

