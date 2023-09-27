import json
import pandas as pd
from datasets import Dataset
import torch

def make_df(path=str):
    with open(path, 'r') as f:
        data= json.load(f)

    df= pd.DataFrame({"questions": [], "queries": [], 'labels': []}, index= [])
    for i in range(len(data)):
        d= data[str(i)]
        df.loc[d['index']]= [d['question'], d['query'], d['verbalization']]
    
    return df

class QAAnswerVerbalizationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item
    
    def __len__(self):
        return len(self.labels['input_ids'])

def prepare_data(tokenizer, train_data, val_data):
    train_data=Dataset.from_pandas(train_data)
    val_data= Dataset.from_pandas(val_data)
    train_data_t= tokenize_data(tokenizer, questions= train_data['questions'], atq= train_data['queries'], labels= train_data['labels'])    
    val_data_t= tokenize_data(tokenizer, questions= val_data['questions'], atq= val_data['queries'], labels= val_data['labels'])
    return train_data_t, val_data_t

def train_val_split(data, frac=0.8):
    """
    Splits the dataframe into Train and Validation dataset.
    data: complete dataframe,
    frac: training data size [0,1]

    returns: pandas dataframes
    -------

    train_data , val_data
    """
    train_data=data.sample(frac=frac,random_state = 42)
    val_data=data.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)
    print('Length Train dataset: ', len(train_data))
    print('Length Val dataset: ', len(val_data))
    return train_data, val_data
    
def tokenize_data(tokenizer, questions, atq, labels):
    encodings = tokenizer(questions, atq, truncation =True, padding =True)
    decodings = tokenizer(labels, truncation= True, padding= True)
    tokenized_data= QAAnswerVerbalizationDataset(encodings, decodings)

    return tokenized_data