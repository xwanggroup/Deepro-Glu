# Deepro-Glu
Title: Deepro-Glu Combination of Convolutional Neural Network and Bi-LSTM Models Using ProtBert and Biological Features to Identify Lysine Glutarylation Sites


Lysine glutarylation (Kglu) is a newly discovered post-translational modification of proteins with important roles in mitochondrial functions, oxidative damage, etc. Established biological experimental methods to identify glutarylation sites are often time-consuming and costly. Therefore, there is an urgent need to develop computational models for efficient and accurate identification of glutarylation sites. In this study, we proposed an ensemble deep learning model which consists of two sub-models. The first sub-model used one-dimensional convolutional neural network(1D-CNN) to capture the protein information from the ProtBert language model, and the second sub-model used bidirectional long short-term memory network (Bi-LSTM) to learn the sequence information from the handcrafted features.


![The Model Architecture](https://github.com/xwanggroup/Deepro-Glu/blob/master/Deepro-Glu.png)
    
### Deepro_Glu uses the following dependencies:
* python 3.6 
* pytorch 
* scikit-learn
* numpy


### Guiding principles:
* Folder 'data' contains train and test data used in this study.
* Folder 'code' contains scripts for training the model and loading data. Load_data.py is the implementation of load raw protein sequences. model.py is the network architecture. Deepro_Glu.py was used to training the proposed model. utils.py is the implementation of calculate model score.
* Folder 'feature_extract' in folder 'code' contains the feature extract method.  aaindex.py is the implementation of AAindex. be_feature.m and exchange_matrix.m are the implementation of BE. BLOSUM62.py is the implementation of BLOSUM62 matrix. DDE.py is the implementation of dipeptide deviation from expected mean. 
* Folder 'result' contains model test result. test.py is the implementation of calculate model test result. 
* You can also run Deepro-Glu model on your own datasets. Just replace the train_example.csv in Deepro_Glu.py with your own dataset.

 
 


#### Note:
This code is for the article 'Deepro-Glu Combination of Convolutional Neural Network and Bi-LSTM Models Using ProtBert and Biological Features to Identify Lysine Glutarylation Sites'.
