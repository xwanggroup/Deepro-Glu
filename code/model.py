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
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoConfig
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
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