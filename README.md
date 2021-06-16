# Privacy-Preserving-Bandits (P2B)
Codes and Data accompanying our paper "[Privacy-Preserving Bandits](https://proceedings.mlsys.org/paper/2020/hash/42a0e188f5033bc65bf8d78622277c4e-Abstract.html)"
```
@inproceedings{malekzadeh2019privacy,
	title        = {Privacy-Preserving Bandits},
	author       = {Malekzadeh, Mohammad and Athanasakis, Dimitrios and Haddadi, Hamed and Livshits, Benjamin},
	year         = {2020},
	booktitle    = {the Third Conference on Machine Learning and Systems (MLSys)}
}
```

Public DOI: [10.5281/zenodo.3685952](https://doi.org/10.5281/zenodo.3685952)

![](https://raw.githubusercontent.com/mmalekzadeh/privacy-preserving-bandits/master/p2b_arch.jpg)

## Note:
* To reproduce the results of the paper, you just need to run codes in the `experiments` folder.
* Multi-Lable datasets will be automatically downloaded for the firs time.
* For criteo dataset, in the first time, use the script `experiments/Criteo/criteo_dataset/create_datasets.ipynb`

### (A) All you need to begin with:
#### 1: Run `1_build_an_encoder.ipynb`.
#### 2: Run `2_a_synthetic_exp.ipynb`.

### (B) For Criteo dataset:
In the directory `experiments/Criteo/`, we have already run this file for the experiment we have reported in Figure 7 and provided dataset by processing `nrows=1000000000`, that uses 1 billion rows of the original dataset.

I If one desires to make a dataset of another `nrows`, for the first time, the script [`create_datasets.ipynb`](https://github.com/mmalekzadeh/privacy-preserving-bandits/tree/master/experiments/Criteo/criteo_dataset) should be used.
You should first set this parameter (number of rows) in the  `create_datasets.ipynb`, build the dataset, and then run the Criteo experiment. Please see [`create_datasets.ipynb`](https://github.com/mmalekzadeh/privacy-preserving-bandits/tree/master/experiments/Criteo/criteo_dataset) for more dtail.


### (C) Info:
You may need to install packages that are listed in the `requirements.txt` file.
 ```
 % pip install -r requirements.txt 
 ```

Specifically, these libraries:
```
%pip install iteround
%pip install pairing 
%pip install scikit-multilearn
%pip install arff
%pip install category_encoders
%pip install matplotlib
%pip install tensorflow
%pip install keras
```
