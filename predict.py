from __future__ import print_function, division
import torch
from my_utils import *
from config import data_transforms
# skorch
from skorch import NeuralNetClassifier
import pickle
import pandas as pd
import argparse
import os

'''
predict using model specified my model_name and save_model
Args: 
    model_name: name of the model, eg: reg34 (available in config.py)
    device: cpu or gpu
    save_model: trained model parameter (must be consistent with model_name, available in config.py)
    testset: pytorch Dataset instance of testing images 
    save_type: save output as csv or pickle? 
    img_path: path of testing images
'''
def predict(model_name, device, save_model, testset, img_path, save_type = "csv"):
    # print("Testing model ", model_name)
    model, input_size = get_pretrained_models(model_name)
    model.eval()
    net = NeuralNetClassifier(model, device = device)
    net.initialize()  # This is important!
    net.load_params(f_params=save_model)
    y_preds = net.predict(testset)
    if save_type == "pickle":
        SAVE_OUTPUT_NAME = model_name + "_outputs.pkl"
        with open(os.path.join("outputs", SAVE_OUTPUT_NAME), "wb") as j:
            pickle.dump({"model_outputs": y_preds}, j)
    elif save_type == "csv":
        SAVE_OUTPUT_NAME = model_name + "_outputs.csv"
        output_df = pd.DataFrame()
        output_df['path'] = img_path
        output_df['label'] = testset.labels
        output_df['output'] = y_preds
        output_df.to_csv(os.path.join("outputs", SAVE_OUTPUT_NAME), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take model name and model path as input')
    parser.add_argument('-m', '--model_name', action="store", help='Name of testing model', required=True)
    parser.add_argument('-p', '--path', action="store", help='Saved parameter path', required=True)
    args = parser.parse_args()
    model, model_param_path = args.model_name, args.path
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    csv_path = "./dataset.csv"
    test_df = pd.read_csv(csv_path)
    test_df['num_label'], _ = pd.factorize(test_df['label'])  # turn labels (string) into numbers
    test_df = test_df[test_df['type'] == "test"]
    img_path = test_df['path'].values
    labels = test_df['num_label'].values
    input_size = 244
    dataset_test = Alzheimer_Dataset(img_path, labels, transform=data_transforms['val'])
    predict(model, device, model_param_path, dataset_test, img_path)
