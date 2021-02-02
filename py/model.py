import torch
from torch import nn

class CBOWClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 output_size,
                 embedding_trainable,
                 dropout,
                 embedding_weight=None):
        super().__init__()
        self.embedding = nn.Embedding(input_size,embedding_size)
        if embedding_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
        if not embedding_trainable:
            self.embedding.weight.requires_grad = False
        self.fc1 = nn.Linear(embedding_size,hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self,input):
        #the first dimention is batch_size
        #size of input:[batch_size,seq_len,embedding_size]
        embedds = self.embedding(input)
        embedds = torch.mean(embedds,dim=1)
        fc1_out = self.dropout(self.relu(self.fc1(embedds)))
        out = self.softmax(self.fc2(fc1_out))
        
        return out

class LSTMClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 dropout,
                 embedding_size,
                 embedding_trainable,
                 bidirectional=True,
                 embedding_weight=None):
        super().__init__()
        self.embedding = nn.Embedding(input_size,embedding_size)
        if embedding_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
        if not embedding_trainable:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout
                            )
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self,input):
        embedds = self.embedding(input)
        out,hidden = self.lstm(embedds)
        mean_out = torch.mean(out,dim=1)
        result = self.softmax(self.fc(mean_out))
        
        return result

#%%
