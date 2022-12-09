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

warnings.filterwarnings('ignore')


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed(777)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    Pre = TP / (TP + FP)

    return SN, SP, ACC, MCC, Pre


def cal_score(pred, label):
    try:
        AUC = roc_auc_score(list(label), pred)
    except:
        AUC = 0

    pred = np.around(pred)
    label = np.array(label)

    confus_matrix = confusion_matrix(label, pred, labels=None, sample_weight=None)
    SN, SP, ACC, MCC, Pre = Model_Evaluate(confus_matrix)
    print("Model score --- SN:{0:.3f}       SP:{1:.3f}       ACC:{2:.3f}       MCC:{3:.3f}      Pre:{4:.3f}   AUC:{5:.3f}".format(SN, SP, ACC, MCC, Pre, AUC))

    return ACC



 
