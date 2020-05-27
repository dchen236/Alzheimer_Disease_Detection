
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
The dataset contains 6400 images of CT scan with 4 classes of labels: {'MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'}

The dataset was obtained from kaggle, 5121 was used for training, 1279 was used for testig, it has imbalance problem, less than 1% images have label - ModerateDemented


### training instruction
training was performed on a GPU with 8000Mib Memory
1D IMAGE gray scale  


### Troubleshoot
If encounter error message: "RuntimeError:  out of memory. Tried to allocate ... "
go to cofig.py and change batchsize into smaller values to fit the memory of your device
