import argparse
import json
import pickle
import torch
import pandas as pd
import numpy as np
from azureml.core.webservice import AciWebservice
from azureml.core.run import Run
from azureml.core import Dataset
from py.create_corpus import Corpus
from py.tokenee import Tokenizer
from torch.utils.data import DataLoader
from torch import nn
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def main():
    parser = argparse.ArgumentError()
    arg = parser.argument_name
    arg('--service_name',type=str)
    arg('--datastore_name',type=str)
    arg('--train_corpus',type=str)
    arg('--eval_corpus',type=str)
    arg('--vocab_path',type=str)
    arg('--frac',type=float,default=0.5)
    arg('--is_sentence',type=str,default='false')
    arg('--max_seq_len',type=int,default=32)
    arg('--eval_batch_size',type=int,default=32)
    arg('--cuda',type=str,default='true')
    args = parser.parse_args()
    size = args.eval_batch_size
    
    run = Run.get_context()
    ws = run.experiment.workspace
        
    datastore = ws.datastores[args.datastore_name]  
    train_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,args.train_corpus)).to_pandas_dataframe()
    eval_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,args.eval_corpus)).to_pandas_dataframe()
    corpus = pd.concat([train_corpus,eval_corpus]).sample(frac=args.frac).reset_index(drop=True)
    vocab = Dataset.File.from_files(path=(datastore,args.vocab_path))
    vocab.download('.',overwrite=True)
    with open('vocab_train.pkl','rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(token_fn=word_tokenize,
                          is_sentence=args.is_sentence.lower(),
                          max_len=args.max_seq_len,
                          vocab=vocab)   
    dataset = Corpus(corpus=corpus,
                 tokenizer=tokenizer,
                 cuda='false')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=size,
                            shuffle=False)
    texts,targets = [],[]
    for text,target in dataloader:
        texts.append(text)
        targets.append(target)
    texts = np.concatenate(texts).astype(np.int32)
    targets = np.concatenate(targets)

    df = []
    for row in texts:
        df.append(row.tolist())
    json_data = json.dumps({'data':df})
    json_data = bytes(json_data,encoding='utf-8')
    
    aci_service = AciWebservice(workspace=ws,
                                name=args.service_name)
    
    try:
        loss_fn = nn.NLLLoss()
        preds = aci_service.run(json_data)
        losses = loss_fn(preds,targets)
        accs = (preds == targets).sum()
        avg_loss,avg_acc = round(losses/len(preds),5),round(accs/len(preds),3)
        output = np.vstack([targets,preds]).transpose()
        output = pd.DataFrame(output,columns=['label','text'])
        output.to_csv('score.csv',index=False)
        datastore.upload_files(files=['score.csv'],
                               target_path='score',
                               overwrite=True)
        metrics = {'score_avg_loss':avg_loss,
                   'score_avg_acc':avg_acc}
        for key,value in metrics.items():
            run.log(key,value)
            run.parent.log(key,value)
        print('successfully saved score result')
    except Exception as error:
        print(error)

if __name__ == '__main__':
    main()

#%%


    