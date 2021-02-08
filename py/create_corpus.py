from torch.utils.data import Dataset
import pandas as pd
import torch

class Corpus(Dataset):
    def __init__(self,
                 corpus,
                 tokenizer,
                 cuda):
        self.tokenizer = tokenizer
        self.cuda = cuda
        self.corpus = corpus
        self.ltoi,self.itol = {},{}
        labels = sorted(self.corpus.label.unique())
        for i,label in enumerate(labels):
            self.ltoi[label] = i
            self.itol[i] = label
        self.corpus['label'] = self.corpus['label'].map(self.ltoi)
    
    def __getitem__(self,index):
        label = self.corpus.loc[index,'label']
        text = self.corpus.loc[index,'text']
        text_tokens = self.tokenizer.tokenize_and_transform(text)
        
        label = torch.tensor(label)
        text_tokens = torch.tensor(text_tokens)
        if self.cuda == 'true':
            label = label.cuda()
            text_tokens = text_tokens.cuda()
        
        return text_tokens,label
    
    def __len__(self):
        return len(self.corpus)
        

#%%
