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


class AQYDataset(Dataset):
    def __init__(self, df, train, label, device):
        self.protein_seq = df

        self.seq_list = train
        self.label_list = label

    def __getitem__(self, index):
        seq = self.protein_seq[index]
        seq_len = len(seq)
        seq = seq.replace('', ' ')
        encoding = tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            max_length=41,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),

        }

        seq_feature = self.seq_list[index]
        label = self.label_list[index]

        return sample, seq_feature, label

    def __len__(self):
        return len(self.protein_seq)


def fit(model, train_loader, optimizer, criterion, device):
    model.train()

    pred_list = []
    label_list = []

    for samples, launch_seq, label in train_loader:
        input_ids = samples['input_ids'].to(device)
        token_type_ids = samples['token_type_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        launch_seq = launch_seq.long().to(device)
        label = torch.tensor(label).float().to(device)
        pred = model(input_ids, token_type_ids, attention_mask, launch_seq)
        pred = pred.squeeze()
        loss = criterion(pred, label)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score


def validate(model, val_loader, device):
    model.eval()

    pred_list = []
    label_list = []

    for samples, launch_seq, label in val_loader:
        input_ids = samples['input_ids'].to(device)
        token_type_ids = samples['token_type_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        launch_seq = launch_seq.long().to(device)
        label = torch.tensor(label).float().to(device)
        pred = model(input_ids, token_type_ids, attention_mask, launch_seq)
        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score


Deepro_train = joblib.load('./Deepro_Glu_train_pred.pkl')
label = joblib.load('./Deepro_Glu_train_label.pkl')


tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert')

class CNN_BiLSTM_Attention(nn.Module):

    def __init__(self, embedding_dim=83, hidden_dim=32, n_layers=1):
        super(CNN_BiLSTM_Attention, self).__init__()

        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
        out_channle = 16
        self.conv1 = nn.Conv1d(1024, out_channle, kernel_size=3, stride=1, padding='same')
        #         self.conv2 = nn.Conv1d(512, 128, kernel_size=3, stride=1, padding='same')
        #         self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding='same')

        self.n_layers = n_layers
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, 64, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(64 * 2, 32, num_layers=n_layers, bidirectional=True, batch_first=True)
        # self.lstm3 = nn.LSTM(2*16, 8, num_layers=n_layers, bidirectional=True, batch_first=True)
        # self.lstm4 = nn.LSTM(2*32, 16, num_layers=n_layers, bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(out_channle + 32 * 2, 16)
        self.fc2 = nn.Linear(16, 5)
        self.fc = nn.Linear(5, 1)

        #         self.batch1 = nn.BatchNorm1d(128)

        self.batch1 = nn.BatchNorm1d(16)

        self.batch2 = nn.BatchNorm1d(128)
        self.batch3 = nn.BatchNorm1d(64)

        self.Rrelu = nn.ReLU()
        self.LRrelu = nn.LeakyReLU()

        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2 + out_channle, hidden_dim * 2 + out_channle))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2 + out_channle, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))

        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score

        context = torch.sum(scored_x, dim=1)
        return context

    def forward(self, input_ids, token_type_ids, attention_mask, x):
        pooled_output, _ = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        imput = pooled_output.permute(0, 2, 1)
        #
        conv1_output = self.conv1(imput)
        #         conv2_output = self.conv2(batch1_output)
        #         batch2_output = self.batch2(conv2_output)
        #         conv3_output = self.conv3(batch2_output)
        #         batch3_output = self.batch3(conv3_output)
        prot_out = torch.mean(conv1_output, axis=2, keepdim=True)
        prot_out = prot_out.permute(0, 2, 1)

        output, (final_hidden_state, final_cell_state) = self.lstm1(x.float())
        output = self.dropout1(output)
        lstmout2, (_, _) = self.lstm2(output)
        bi_lstm_output = self.dropout2(lstmout2)
        bi_lstm_output = torch.mean(bi_lstm_output, axis=1, keepdim=True)

        fusion_output = torch.cat([prot_out, bi_lstm_output], axis=2)

        attn_output = self.attention_net(fusion_output)

        out1 = self.fc1(attn_output)
        #         out1 = self.Rrelu(out1)
        out2 = self.fc2(out1)
        #         out1 = self.Rrelu(out2)
        logit = self.fc(out2)

        return nn.Sigmoid()(logit)


train_score = cal_score(Deepro_train, label)


