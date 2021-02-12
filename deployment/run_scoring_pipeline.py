import sys
sys.path.append('azure')
from env_variables import ENV
from azureml.core import Workspace,Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.pipeline.core import PublishedPipeline
import json

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
    
    matched_pipeline = []
    pipeline_list = PublishedPipeline.list(ws)
    for pipe in pipeline_list:
        if pipe.name == env.pipeline_scoring_name and pipe.version == env.build_id:
            matched_pipeline.append(pipe)
    if len(matched_pipeline) > 1:
        raise Exception('there should be only one matched pipeline')
    elif len(matched_pipeline) == 0:
        raise Exception('no pipeline matched')
    else:
        published_pipeline = matched_pipeline[0]
    
    tags = {'description':'news category classifiction experiment'}
    if env.build_id is not None:
        tags.update({'build_id':env.build_id})
        if env.build_url is not None:
            tags.update({'build_url':env.build_url})
        
    exp = Experiment(workspace=ws,
                     name=env.experiment_scoring_name)
    exp.submit(published_pipeline,
               tags=tags,
               pipeline_parameters={
                                    'model_name':env.model_name,
                                    'model_version':env.model_version,
                                    'service_name':env.service_name,
                                    'env_name':env.environment_name,
                                    'train_corpus':'corpus/corpus_train.csv',
                                    'eval_corpus':'corpus/corpus_eval.csv',
                                    'frac':env.corpus_frac,
                                    'datastore_name':env.datastore_name,
                                    'is_sentence':env.is_sentence,
                                    'max_len':env.max_seq_len,
                                    'batch_size':env.batch_size_eval_scoring,
                                    'cuda':env.cuda,
                                    'vm_size_scoring':env.vm_size_scoring
                                     })
    print(f'pipeline {env.pipeline_scoring_name} has been built')

if __name__ == '__main__':
    main()

#%%
#dc = {'a':1,'b':3,'c':5,'d':7,'e':9}
#def power(input):
#    return input**2
#def trail(dc):
#    for key,value in dc.items():
#        yield(power(value))
#for v in trail(dc):
#    print(v+3)