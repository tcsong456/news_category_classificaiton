from azureml.core.model import Model,Dataset
from azureml.core.run import Run
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from py.create_corpus import Corpus
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from azure_utils import use_or_create_datastore
import pickle
import sys
sys.path.append('py')
from tokenee import Tokenizer

def get_model(ws,
              model_name,
              model_version=None,
              tags=None):
    model_version = int(model_version)
    if model_version >= 0:
        model = Model(workspace=ws,
                      name=model_name,
                      version=model_version,
                      tags=tags)
    else:
        models = Model.list(workspace=ws,
                            name=model_name,
                            latest=True)
        model = models[-1]
    
    if model is None:
        raise FileNotFoundError('model not found')
    
    return model

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_name',type=str,default='news_clf_model.pt',
        help='name of model')
    arg('--model_version',type=str,
        help='the version of model')
    arg('--tag_name',type=str,
        help='the name of tag')
    arg('--tag_value',type=str,
        help='the value of tag')
    arg('--dataset',type=str,
        help='dataset used to build dataloader')
    arg('--cuda',type=str,default='true',
        help='if gpu is enabled')
    arg('--batch_size',type=int,default=16,
        help='batch size for evaluation')
    arg('--vocab_path',type=str,
        help='the path to vocab file')
    arg('--is_sentence',type=str,default='false')
    arg('--max_seq_len',type=int,default=128)
    arg('--train_corpus',type=str,
        help='the path to training corpus')
    arg('--eval_corpus',type=str,
        help='the path to evaluation corpus')
    arg('--corpus_frac',type=str,
        help='the fraction of corpus to be used for eval')
    args = parser.parse_args()
    
    run = Run.get_context()
    ws = run.experiment.workspace
    
    tags = None
    if args.tag_name is not None and args.tag_value is not None:
        tags = [(args.tag_name,args.tag_value)]
    
    model = get_model(ws=ws,
                      model_name=args.model_name,
                      model_version=args.model_version,
                      tags=tags)
    model_path = Model.get_model_path(model_name=model.name,
                                      version=model.version,
                                      _workspace=ws)
    model = torch.load(model_path)
    model = model.cuda() if args.cuda=='true' else model
    model.eval()
    
    datastore = ws.datastores['news_cat_clf']
    train_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,args.train_corpus)).to_pandas_dataframe()
    eval_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,args.eval_corpus)).to_pandas_dataframe()
    corpus = pd.concat([train_corpus,eval_corpus]).sample(frac=args.corpus_frac).reset_index(drop=True)
    tokenize_fn = word_tokenize
    vocab = Dataset.File.from_files(path=(datastore,args.vocab_path))
    vocab.download('.',overwrite=True)
    vocab_path = args.vocab_path.split('/')[-1]
    with open(vocab_path,'rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(token_fn=tokenize_fn,
                          is_sentence=args.is_sentence,
                          max_len=args.max_seq_len,
                          vocab=vocab)
    dataset = Corpus(corpus=corpus,
                     tokenizer=tokenizer,
                     cuda=args.cuda)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    
    losses,accs = 0,0
    results,targets = [],[]
    loss_fn = nn.NLLLoss()
    for texts,target in dataloader:
        preds = model(texts)
        loss = loss_fn(preds,target)
        losses += loss.item()
        result = np.argmax(preds.cpu().data,axis=-1)
        target = target.cpu().data
        results.append(result.numpy()),targets.append(target.numpy())
        acc = (result == target).sum()
        accs += acc
    
    n_samples = len(dataloader.dataset)
    n_rounds = np.ceil(n_samples / args.batch_size)
    avg_losses = np.round(losses / n_rounds,5)
    avg_acc = np.round(accs / n_samples,3)
    metrics = {'batchscore_avg_losses':avg_losses,
               'batchscore_avg_acc':avg_acc}
    for key,value in metrics.items():
        run.log(key,value)
        run.parent.log(key,value)
    results = np.concatenate(results)
    targets = np.concatenate(targets)
    output = np.vstack([np.array(targets),np.array(results)]).transpose()
    output = pd.DataFrame(output,columns=['label','pred'])
    output.to_csv('batch_socre.csv',index=False)
    
    datastore = use_or_create_datastore(ws=ws,
                                        datastore_name='news_cat_clf',
                                        default=False)
    datastore.upload_files(files=['batch_socre.csv'],
                           target_path='batchscore',
                           overwrite=True)
    

if __name__ == '__main__':
    main()

#%%
