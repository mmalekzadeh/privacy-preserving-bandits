# Privacy-Preserving-Bandits (P2B)
Codes and Data accompanying our paper "[Privacy-Preserving Bandits](https://arxiv.org/pdf/1909.04421.pdf
)"
```
Mohammad Malekzadeh, Dimitrios Athanasakis, Hamed Haddadi, and Benjamin Livshits,
"Privacy-Preserving Bandits", 
Accepted for the Third Conference on Machine Learning and Systems (MLSys 2020), March, 2020.
```
## Note:
* To reproduce the results of the paper, you just need to run codes in the `experiments` folder.
* Multi-Lable datasets will be automatically downloaded for the firs time.
* For criteo dataset, in the first time, use the script `experiments/Criteo/criteo_dataset/create_datasets.ipynb`

### All you need to begin with:
#### 1: Run `1_build_an_encoder.ipynb`.
#### 2: Run `2_a_synthetic_exp.ipynb`.



### Info:
If you don't run `1_build_an_encoder.ipynb`, you may need to install these libraries:
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
