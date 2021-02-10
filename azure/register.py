from azureml.core.model import Model
from azureml.core.run import Run
import torch
import argparse
import json
import os
import sys
sys.path.append('py')

def register_model(run_id,
                   exp,
                   model_path,
                   model_name,
                   mtags=None):
    try:
        tags = {'title':'news_classification',
                'run_id':run_id,
                'exp_name':exp.name}
        if mtags is not None:
            tags.update(mtags)
        
        Model.register(workspace=exp.workspace,
                       model_path=model_path,
                       model_name=model_name,
                       tags=tags)
        print(f'model:{model_name} has been sucessfully registered')
    except Exception as error:
        print(error)
        exit(1)

def fix(model,datastore,):
    import pandas as pd
    import pickle
    from azureml.core import Dataset
    import nltk
    nltk.download('punkt')
    from create_corpus import Corpus
    from tokenee import Tokenizer
    from torch.utils.data import DataLoader
    from torch import nn
    import numpy as np
    from nltk.tokenize import word_tokenize
    vocab = Dataset.File.from_files(path=(datastore,'corpus/vocab_train.pkl'))
    vocab.download('.',overwrite=True)
    vocab_path = 'corpus/vocab_train.pkl'.split('/')[-1]
    with open(vocab_path,'rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(token_fn=word_tokenize,
                          is_sentence='false',
                          max_len=64,
                          vocab=vocab)
    losses,accs = 0,0
    train_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,'corpus/corpus_train.csv')).to_pandas_dataframe()
    eval_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,'corpus/corpus_eval.csv')).to_pandas_dataframe()
    corpus = pd.concat([train_corpus,eval_corpus]).sample(frac=0.4).reset_index(drop=True)  
    dataset = Corpus(
                     corpus,
                     tokenizer,
                     'true')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=64,
                            shuffle=False)
    
    results,targets = [],[]
    loss_fn = nn.NLLLoss()
    n_samples = len(dataloader.dataset)
    n_rounds = np.ceil(n_samples / 64)
    model = model.cuda()
    model.eval()
    for texts,target in dataloader:
        preds = model(texts)
        loss = loss_fn(preds,target)
        losses += loss.item()
        result = np.argmax(preds.cpu().data,axis=-1)
        target = target.cpu().data
        results.append(result.numpy()),targets.append(target.numpy())
        acc = (result == target).sum()
        accs += acc.item()
    avg_losses = np.round(losses / n_rounds,5)
    avg_acc = np.round(accs / n_samples,3)
    print(f'avg losses: {avg_losses},avg acc: {avg_acc}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str)
    parser.add_argument('--input',type=str)
    args = parser.parse_args()
    
    run = Run.get_context()
    exp = run.experiment
    ws = exp.workspace
    
    with open('config.json','r') as f:
        config = json.load(f)
    register_tags = config['registration']
    mtags = {}
    print(run.parent.get_metrics())
    for tag in register_tags['tags']:
        try:
            if tag == 'eval_avg_acc':
                value = max(run.parent.get_metrics()[tag])
            else:
                value = run.parent.get_metrics()[tag]
            mtags[tag] = value
        except KeyError:
            print(f'{tag} key not found')
    
    model_path = os.path.join(args.input,args.model_name)
    model = torch.load(model_path)
    
    if model is not None:
        run_id = run.id
        register_model(run_id=run_id,
                       exp=exp,
                       model_path=model_path,
                       model_name=args.model_name,
                       mtags=mtags)
    else:
        raise Exception('model is not found')
    
    datastore = ws.datastores['news_cat_clf']
    models = Model.list(workspace=ws,
                        name=args.model_name,
                        latest=True)
    model = models[-1]
    model_path = Model.get_model_path(model_name=model.name,
                                      version=model.version,
                                      _workspace=ws)
    model = torch.load(model_path)
    fix(model,datastore)
            

if __name__ == '__main__':
    main()
#%%
