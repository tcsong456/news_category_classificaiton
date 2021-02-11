import sys
sys.path.append('azure')
import json
from azure_utils import *
from env_variables import ENV
from azureml.core import RunConfiguration
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineParameter,Pipeline

def main():
    env = ENV()
    with open('config.json','r') as f:
        config = json.load(f)        
    auth = ServicePrincipalAuthentication(tenant_id=config['tenant_id'],
                                          service_principal_id=config['service_principal_id'],
                                          service_principal_password=config['service_principal_password'])
    ws = use_or_create_workspace(workspace_name=env.workspace,
                                 resource_group=env.resource_group,
                                 subscription_id=env.subscription_id,
                                 auth=auth) 
    gpu_compute_target = use_or_create_computetarget(ws=ws,
                                                     e=env,
                                                     cpu_cluster=False,
                                                     compute_name='gpu-cluster',
                                                     batch_scoring=True)
    environment = use_or_create_environment(ws=ws,
                                            env_name=env.environment_name,
                                            conda_dependencies='yaml/run_dependencies.yml',
                                            create_new_env=True,
                                            overwrite=True)
    
    runconfig = RunConfiguration()
    runconfig.environment = environment
    
    model_name_param = PipelineParameter('model_name',default_value='news_clf_model.pt')
    model_version_param = PipelineParameter('model_version',default_value='0')
    service_name_param = PipelineParameter('service_name',default_value='news_clf_aci')
    env_name_param = PipelineParameter('env_name',default_value='nes_env')
    train_corpus_param = PipelineParameter('train_corpus',default_value='train_corpus')
    eval_corpus_param = PipelineParameter('eval_corpus',default_value='eval_corpus')
    frac_param = PipelineParameter('frac',default_value=0.4)
    datastore_name_param = PipelineParameter('datastore_name',default_valule='news_cat_clf')
    
    deploy_step = PythonScriptStep(name='deploy_step',
                                   script_name='deployment/deploy.py',
                                   source_directory='.',
                                   arguments=['--model_name',model_name_param,
                                              '--model_version',model_version_param,
                                              '--service_name',service_name_param,
                                              '--env_name',env_name_param],
                                   runconfig=runconfig,
                                   compute_target=gpu_compute_target,
                                   allow_reuse=False)
    print('deploy step is built')
    
    test_step = PythonScriptStep(name='test_step',
                                   script_name='deployment/aci_test.py',
                                   source_directory='.',
                                   arguments=[
                                              '--service_name',service_name_param,
                                              '--env_name',env_name_param,
                                              '--train_corpus',train_corpus_param,
                                              '--eval_corpus',eval_corpus_param,
                                              '--frac',frac_param,
                                              '--datastore_name',datastore_name_param
                                              ],
                                   runconfig=runconfig,
                                   compute_target=gpu_compute_target,
                                   allow_reuse=False)
    print('test step is built')
    test_step.run_after(deploy_step)
    
    pipeline = Pipeline(workspace=ws,
                        steps=[deploy_step,test_step])
    pipeline.publish(name=env.pipeline_scoring_name,
                     description='news clf scoring pipeline',
                     version=env.build_id)
    
    

#%%