### TODO:
- write training instructions
- upload models to googleDrive
- create activation map / heapmap on image
- sample image from each class 

### Dependencies

Python 3.7.3
install pytorch at : https://pytorch.org/
```
pip install pandas
pip install skorch 
pip install seaborn
pip install scikit-learn
```

### Dataset
The dataset contains 6400 images of MRI with 4 classes of labels: {'MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'}

The dataset was obtained from [kaggle](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images), 5121 was used for training, 1279 was used for testig, it has imbalance problem, less than 1% images have label - ModerateDemented

![](https://github.com/dchen236/Alzheimer_Disease_Detection/blob/master/figures/imbalance.png)


### Experiments and Result
The results can be viewed from report_performance.ipynb as well.

We have trained 9 models including: resnet18, resnet34, resnet50, resnet101, resnet152, squeezenet, VGG, alexNet and densenet

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

### training instruction
training was performed on a GPU with 8000Mib Memory
1D IMAGE gray scale  


### Troubleshoot
If encounter error message: "RuntimeError:  out of memory. Tried to allocate ... "
go to cofig.py and change batchsize into smaller values to fit the memory of your device
