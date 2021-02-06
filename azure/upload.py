from azureml.core import Workspace
from env_variables import ENV
from azureml.core.authentication import ServicePrincipalAuthentication
import json
import pandas as pd
from azure_utils import use_or_create_datastore
import argparse

def parserargs():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--corpus_train',type=str,
        help='the path to train corpus')
    arg('--corpus_eval',type=str,
        help='the path to eval corpus')
    arg('--vocab',type=str,
        help='the path to vocab.pkl')
    args = parser.parse_args()
    return args

def dataframe(datapath:str,name:str):
    corpus = []
    with open(datapath,'r') as f:
        for line in f:
            _line = line.split('\t')
            label,text = _line[0],' '.join(_line[1:]).strip()
            corpus.append([label,text])
    corpus = pd.DataFrame(corpus,columns=['label','text'])
    corpus.to_csv(name,index=False)

def main():
    args = parserargs()
    env = ENV
    with open('config.json','r') as f:
        config = json.load(f)        
    auth = ServicePrincipalAuthentication(tenant_id=config['tenant_id'],
                                          service_principal_id=config['service_principal_id'],
                                          service_principal_password=config['service_principal_password'])
    ws = Workspace.get(name=env.workspace,
                       resource_group=env.resource_group,
                       subscription_id=env.subscription_id,
                       auth=auth)
    datastore = use_or_create_datastore(ws=ws,
                                        datastore_name=env.datastore_name)
    dataframe(args.corpus_train,'corpus_train.csv')
    dataframe(args.corpus_eval,'corpus_eval.csv')
    
    for input in ['corpus_train.csv','corpus_eval.csv',args.vocab]:
        datastore.upload_files([input],
                               target_path='corpus',
                               overwrite=True)
            
if __name__ == '__main__':
    main()
        
    
#%%
