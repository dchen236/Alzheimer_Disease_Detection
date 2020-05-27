dependencies:
pandas
skorch
pytorch


1D IMAGE gray scale  

dataset spec : train size : 6400 test size : ...


If encounter error message: 
RuntimeError:  out of memory. Tried to allocate ... 
go to cofig.py and change batchsize into smaller values to fit the memory of your device
