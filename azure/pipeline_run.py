from azureml.core import Workspace,Experiment
from azureml.pipeline.core import PublishedPipeline
from azureml.core.authentication import ServicePrincipalAuthentication
from env_variables import ENV
import json
#import argparse

#def parseargs():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--bidirectional',action='store_true',
#                        help='if bidirectional network is used')
#    args = parser.parse_args()
#    return args

def main():
    env = ENV()
    with open('config.json','r') as f:
        config = json.load(f)
    auth = ServicePrincipalAuthentication(tenant_id=config['tenant_id'],
                                          service_principal_id=config['service_principal_id'],
                                          service_principal_password=config['service_principal_password'])
    ws = Workspace.get(name=env.workspace,
                       resource_group=env.resource_group,
                       subscription_id=env.subscription_id,
                       auth=auth)
    
    matched_pipe = []
    pipeline_list = PublishedPipeline.list(ws)
    for pipe in pipeline_list:
        if pipe.name == env.pipeline_name and pipe.version == env.build_id:
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
                     name=env.experiment_name)
    exp.submit(published_pipeline,
               tags=tags,
               pipeline_parameters={'model_name':env.model_name,
                                    'model_version':env.model_version,
                                    'cuda':env.cuda,
                                    'batch_train':env.batch_size_train,
                                    'batch_eval':env.batch_size_eval,
                                    'tokenizer':env.tokenizer,
                                    'vocab':'corpus/vocab_train.pkl',
                                    'is_sentence':env.is_sentence,
                                    'max_seq_len':env.max_seq_len,
                                    'train_corpus':'corpus/corpus_train.csv',
                                    'eval_corpus':'corpus/corpus_eval.csv',
                                    'mode':env.model_mode,
                                    'hidden_size':env.hidden_size,
                                    'num_layers':env.num_layers,
                                    'dropout':env.dropout,
                                    'embedding_size':env.embedding_size,
                                    'embedding_trainable':env.embedding_trainable,
                                    'use_word_embedding':env.use_word_embedding,
                                    'learning_rate':env.learning_rate,
                                    'epochs':env.epochs,
                                    })
    print(f'{env.pipeline_name} has been submitted!')

if __name__ == '__main__':
    main()
#%%
