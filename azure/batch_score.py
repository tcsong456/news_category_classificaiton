from azureml.core.model import Model
from azureml.core.run import Run
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from py.create_corpus import Corpus
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from azure_utils import use_or_create_datastore

def get_model(ws,
              model_name,
              model_version=None,
              tags=None):
    if model_version:
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
    arg('--cuda',action='store_true',
        help='if gpu is enabled')
    arg('--batch_size',type=int,defautl=16,
        help='batch size for evaluation')
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
    model = model.cuda() if args.cuda else model
    model.eval()
    
    tokenizer_fn = word_tokenize
    dataset = Corpus(corpus_path=args.dataset,
                     tokenizer=tokenizer_fn,
                     cuda=args.cuda)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    
    losses,accs = 0,0
    results,targets = [],[]
    loss_fn = nn.NLLLoss()
    for texts,target in dataloader:
        preds = model.predict(texts)
        loss = loss_fn(preds,target)
        losses += loss.item()
        result = np.argmax(preds.cpu().data,axis=-1)
        target = target.cpu().data
        results.append(result),targets.append(target)
        acc = (result == target).sum()
        accs += acc
    
    n_rounds = np.ceil(len(dataloader) / args.batch_size)
    avg_losses = np.round(losses / n_rounds,5)
    avg_acc = np.round(accs / len(dataloader),3)
    metrics = {'batchscore_avg_losses':avg_losses,
               'batchscore_avg_acc':avg_acc}
    for key,value in metrics:
        run.log(key,value)
        run.parent.log(key,value)
    output = np.vstack([np.array(result),np.array(target)]).transpose()
    output = pd.DataFrame(output,columns=['pred','target'])
    
    datastore = use_or_create_datastore(ws=ws,
                                        datastore_name='news_cat_clf')
    datastore.upload_files(files=['batch_socre.csv'],
                           target_path='batchscore',
                           overwrite=True)
    

if __name__ == '__main__':
    main()

#%%