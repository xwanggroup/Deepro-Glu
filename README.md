# Deepro-Glu
Title: Deepro-Glu Combination of Convolutional Neural Network and Bi-LSTM Models Using ProtBert and Biological Features to Identify Lysine Glutarylation Sites


Lysine glutarylation (Kglu) is a newly discovered post-translational modification of proteins with important roles in mitochondrial functions, oxidative damage, etc. Established biological experimental methods to identify glutarylation sites are often time-consuming and costly. Therefore, there is an urgent need to develop computational models for efficient and accurate identification of glutarylation sites. In this study, we proposed an ensemble deep learning model which consists of two sub-models. The first sub-model used one-dimensional convolutional neural network(1D-CNN) to capture the protein information from the ProtBert language model, and the second sub-model used bidirectional long short-term memory network (Bi-LSTM) to learn the sequence information from the handcrafted features.


![The Model Architecture](https://github.com/zydingg/Deepro-Glu/blob/main/Deepro-Glu.png)
    
### Deepro_Glu uses the following dependencies:
* python 3.6 
* pytorch 
* scikit-learn
* numpy


### Guiding principles:
* File 'data' contains training and testing data used in this study.
* File 'code' contains scripts for training the model and loading data. Load_data.py is the implementation of load raw protein sequence. Deepro_Glu.py was used to training the proposed model. utils.py is the implementation of calculate training model score.
* File 'result' contains model test result. test.py is the implementation of calculate model test result. 






#### Note:
This code is for the article 'Deepro-Glu Combination of Convolutional Neural Network and Bi-LSTM Models Using ProtBert and Biological Features to Identify Lysine Glutarylation Sites'.
