import sys
sys.path.append('azure')
import json
import pickle
import pandas as pd
import numpy as np
from azureml.core.webservice import AciWebservice
from azureml.core.run import Run
from azureml.core import Dataset
from py.create_corpus import Corpus
from py.tokenee import Tokenizer
from torch.utils.data import DataLoader
from env_variables import ENV
from torch import nn
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def main():
    env = ENV()
    size = env.scoring_batch_size
    run = Run.get_context()
    ws = run.experiment.workspace
        
    datastore = ws.datastores[env.datastore_name]  
    train_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,env.train_corpus)).to_pandas_dataframe()
    eval_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,env.eval_corpus)).to_pandas_dataframe()
    corpus = pd.concat([train_corpus,eval_corpus]).sample(frac=env.corpus_frac).reset_index(drop=True)
    vocab = Dataset.File.from_files(path=(datastore,env.vocab_path))
    vocab.download('.',overwrite=True)
    with open(env.vocab_path.split('/')[-1],'rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(token_fn=word_tokenize,
                          is_sentence=env.is_sentence.lower(),
                          max_len=env.max_seq_len,
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
                                name=env.service_name)
    
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


    