### TODO:
- upload models to googleDrive
- add references

### Table of content

- [Dependency](#Dependency)
- [Dataset](#Dataset)
- [Experiments and Result](#Experiments-and-Result)
  * [training](#training)
  * [testing performance](#multiclass-performance)
- [Training instruction](#Training-instruction)
- [Troubleshoot](#Troubleshoot)
- [Resources](#Resources)


### Dependency

Python 3.7.3
install pytorch at : https://pytorch.org/
```
pip install pandas
pip install skorch 
pip install seaborn
pip install scikit-learn
pip install tqdm
```

### Dataset
The dataset contains 6400 images of MRI with 4 classes of labels: {'MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'}

The dataset was obtained from [kaggle](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images), 5121 was used for training, 1279 was used for testig, it has imbalance problem, less than 1% images have label - ModerateDemented

![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/imbalance.png)


### Experiments and Result

The results can be viewed from [report_performance.ipynb](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/report_performance.ipynb) as well.

- We have trained 9 models including: resnet18, resnet34, resnet50, resnet101, resnet152, squeezenet, VGG, alexNet and densenet
- We use 10% of training set for validation (stratified on 4 classes, important to do so as the dataset has imbalance problem)

#### training
During training: most of the classes were able to achieve 99% validation accuracy except for squeezenet (around 56%)
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/sample_train_loss.png)
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/train_loss_squeeze.png)

#### multiclass performance
multi_class_performance was measured weighted one vs all metrics, for instance, to measure class VeryMildDemented, we treat the rest of 3 classes (MildDemented, ModerateDemented, NonDemented) as Not VeryMildDemented. Using this evaluation, we measure the accuracy for each of the 4 classes, then taking average.
 [More detailed explanation can be viewed from sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules), we used average = "weighted".
 
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/multi_class_performance.png)
 
 #### Binary performance
 Another approach we used to treat classes as {Demented, Not_Demented} (Demented includes all 3 level of Dementia).
 Then the accuracy can be evaluated using standard binary classfication sestting. 
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/binary_performance.png)

#### Samples of Confusion Matrix
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/confusion_matrix_res50.png)
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/confusion_matrix_res152.png)
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/confusion_matrix_vgg.png)
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/confusion_matrix_dense.png)

#### Grad-Cam Activation
we have compared the actual testing image with grad-cam activation overlayed images using resnet models, more results can be viewed at [Grad-cAM.ipynb](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/Grad-cAM.ipynb)

![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/grad_cam10_res34.png)
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/grad_cam46_res34.png)
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/grad_cam80_res34.png)
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/grad_cam10_res152.png)
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/grad_cam_46_res152.png)
![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/grad_cam100_res152.png)

### Training instruction
Training was performed on a GPU with 8000Mib Memory (GPU is not required, but it will be slow during training and testing)
- make sure you have installed the dependency from [Dependency](#Dependency) section. 
- obtain dataset from [kaggle](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images) and save at the folder as train.py
- download [dataset.csv](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/dataset.csv) from this repo, save at the same folder as train.py, this csv file
- to train all 9 models, nothing needs to be modified, rum train.py using python3 train.py
- to train with selected models, go to config.py and modify `MODELS`
- if you encounter memory issues during training, go to config.py modify batch size for the model causing the memory error
- the default max epoch is 50 with early stopping, patience level 5, threshold 0.01 (stop training if validatin accuracy didn't improve for more than 0.01 after 5 epochs)
- the trained model will be saved at folder models which can be then used for prediction with predict.py
- the training history is saved as json files stored in train_histories including training loss, validation loss and validation accuracy for each epoch and each batch.


### Troubleshoot
If encounter error message: "RuntimeError:  out of memory. Tried to allocate ... "
go to cofig.py and change batchsize into smaller values to fit the memory of your device

### Resources
