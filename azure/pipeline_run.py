from azureml.core import Workspace,Experiment,Dataset
from azureml.pipeline.core import PublishedPipeline
from azureml.core.authentication import ServicePrincipalAuthentication
from env_variables import ENV
import json
import pickle
import argparse
import sys
sys.path.append('azure')
from azure_utils import use_or_create_datastore

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab',help='built vocabulary to be used')
    parser.add_argument('--train_corpus',help='training data to be used')
    parser.add_argument('--eval_corpus',help='evaluation data to be used')
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    env = ENV()
    with open('config.json','r') as f:
        config = json.load(f)
    auth = ServicePrincipalAuthentication(tenant_id=config['tenant_id'],
                                          service_principal_id=config['service_principal_id'],
                                          service_principal_password=config['service_principal_password'])
    ws = Workspace.get(name=env.workspace,
                       resource_grouop=env.resource_group,
                       subscription_id=env.subscription_id,
                       auth=auth)
    datastore = use_or_create_datastore(ws=ws,
                                        datastore_name=env.datastore_name,
                                        use_default=False)
    train_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,args.train_corpus))
    eval_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,args.eval_corpus))
    vocab = Dataset.file.from_files(path=(datastore,args.vocab))
    vocab.download('.',overwrite=True)
    with open(env.vocab,'r') as f:
        vocab = pickle.load(f)
    
    matched_pipe = []
    pipeline_list = PublishedPipeline.list(ws)
    for pipe in pipeline_list:
        if pipe.name == env.pipeline_name and pipe.version.env.build_id:
            matched_pipe.append(pipe)
    if len(matched_pipe) > 1:
        raise ValueError('there should be only one matched pipeline')
    elif len(matched_pipe) == 0:
        raise ValueError('no pipeline matched')
    else:
        published_pipeline = matched_pipe[0]
    
    tags = {}
    if env.build_id is not None:
        tags.update({'build_id':env.build_id})
        if env.build_url is not None:
            tags.update({'build_url':env.build_url})
    exp = Experiment(workspace=ws,
                     name=env.workspace_name)
    exp.submit(published_pipeline,
               tags=tags,
               pipeline_parameters={'model_name':env.model_name,
                                    'model_version':env.model_version,
                                    'cuda':env.cuda,
                                    'batch_train':env.batch_size_train,
                                    'batch_eval':env.batch_size_eval,
                                    'tokenizer':env.tokenizer,
                                    'vocab':vocab,
                                    'is_sentence':env.is_sentence,
                                    'max_seq_len':env.max_seq_len,
                                    'train_corpus':train_corpus,
                                    'eval_corpus':eval_corpus,
                                    'mode':env.fasttext_mode,
                                    'hidden_size':env.hidden_size,
                                    'num_layers':env.num_layers,
                                    'dropout':env.dropout,
                                    'embedding_size':env.embedding_size,
                                    'embedding_trainable':env.embedding_trainable,
                                    'use_word_embedding':env.use_word_embedding,
                                    'bidirectional':env.bidirectional,
                                    'learning_rate':env.learning_rate,
                                    'epochs':env.epochs,
                                    })
    print(f'{env.pipeline_name} has been submitted!')

if __name__ == '__main__':
    main()
#%%
