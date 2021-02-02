from torch.utils.data import Dataset
import pandas as pd
import torch

class Corpus(Dataset):
    def __init__(self,
                 corpus_path,
                 tokenizer,
                 cuda):
        self.tokenizer = tokenizer
        self.cuda = cuda
        self.corpus = []
        self.ltoi,self.itol = {},{}
        with open(corpus_path,'r',encoding='utf8',errors='ignore') as f:
            for line in f:
                _line = line.split('\t')
                self.corpus.append([_line[0],' '.join(_line[1:]).strip()])
        self.corpus = pd.DataFrame(self.corpus,columns=['label','text'])
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
        if self.cuda:
            label = label.cuda()
            text_tokens = text_tokens.cuda()
        
        return text_tokens,label
    
    def __len__(self):
        return len(self.corpus)
        

#%%