import sys
sys.path.append('py')
from torch.utils.data import DataLoader
from torch import nn
from py.create_corpus import Corpus
from py.tokenee import Tokenizer
from azureml.core import Model,Dataset
from azureml.core.run import Run
import numpy as np
import sys
sys.path.append('py')
import pandas as pd
import os
import torch
import json
import pickle
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def init():
    global model
    model_path = os.environ.get('AZUREML_MODEL_DIR')
    print(model_path)
    model_path = Model.get_model_path(model_path.split('/')[-2])
    model = torch.load(model_path)

def run(data):
    batch_size = 64
    run = Run.get_context()
    ws = run.experiment.workspace
    datastore = ws.datastores['news_cat_clf']
    vocab = Dataset.File.from_files(path=(datastore,'corpus/vocab_train.pkl'))
    vocab.download('.',overwrite=True)
    with open('vocab_train.pkl','rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(token_fn=word_tokenize,
                          is_sentence='false',
                          max_len=64,
                          vocab=vocab)    
    
    raw_data = json.loads(data['data'])
    raw_data = pd.DataFrame(raw_data,columns=['label','text'])
    dataset = Corpus(corpus=raw_data,
                     tokenizer=tokenizer,
                     cuda='true')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False)
    
    loss_fn = nn.NLLLoss()
    n_samples = len(dataloader.dataset)
    n_rounds = np.ceil(n_samples / batch_size)
    losses,accs = 0,0
    result = []
    for text,target in dataloader:
        pred = model(text)
        loss = loss_fn(pred,target)  
        losses += loss.item()
        pred = pred.cpu().data
        target = target.cpu().data
        output = np.argmax(pred,axis=-1)
        acc = (output == target).sum()
        accs += acc
        result.append([target,pred])
    avg_acc = np.round(accs / n_samples,3)
    avg_loss = np.round(losses / n_rounds,5)
    result = pd.DataFrame(result,columns=['lable','text'])
    
    return avg_acc,avg_loss,result