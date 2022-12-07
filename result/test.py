#model test
import torch
import torch.nn as nn
import os
import re
import copy
import joblib
import random
import math
import warnings
from torch.utils.data import DataLoader, TensorDataset, Dataset
import warnings
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoConfig
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold

def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    

    return SN, SP, ACC, MCC


def call_score(pred, label):

    pred = np.around(pred)
    label = np.array(label)

    confus_matrix = confusion_matrix(label, pred, labels=None, sample_weight=None)
    SN, SP, ACC, MCC = Model_Evaluate(confus_matrix)
    print("Model score --- SN:{0:.3f}       SP:{1:.3f}       ACC:{2:.3f}       MCC:{3:.3f}   ".format(SN, SP, ACC, MCC))

    return ACC



test_pred = joblib.load('./DeeproGlu_Independtest_pred.pkl')
test_label = joblib.load('./DeeproGlu_Independtest_label.pkl')

train_score = call_score(test_pred,test_label)