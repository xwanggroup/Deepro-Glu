import pandas as pd
import numpy as np

def AAindex(sequence):
    obj = pd.read_csv('./AAindex_12.csv')  
    pro_name_list = obj['AccNo'].tolist()

    AA_list_sort = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    AAindex_dict = {}
    for ele in AA_list_sort:   
        AAindex_dict[ele] = obj[ele].tolist()   
    AAindex_dict['X'] = [0] * 12  
    feature_vector = []  
    for item in sequence:   
        feature_vector.extend(AAindex_dict[item])
    return feature_vector


data_pos = pd.read_excel('row_pos.xlsx')
data_neg = pd.read_excel('row_neg.xlsx')
data_pos.columns = [ 'Sequences', 'Label']
data_neg.columns = [ 'Sequences', 'Label']


def feature_extract(data):
    train = data.Sequences.values
    train_Label = data.Label.values
    pd_data = []
    for i, seq in enumerate(train):
        seq_feature = [seq, train_Label[i]]
        seq_feature += AAindex(seq) 
        pd_data.append(seq_feature)
    pd_data = pd.DataFrame(pd_data)
    return pd_data


data_positive = feature_extract(data_pos)
data_positive.head()
data_positive.to_csv('data_pos_aaindex.csv', index=False)


data_negative = feature_extract(data_neg)
data_negative.head()
data_negative.to_csv('data_neg_aaindex.csv', index=False) 
