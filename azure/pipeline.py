from azure_utils import *
from azureml.core import RunConfiguration
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.pipeline.steps import PythonScriptStep
from azureml.data.datapath import DataPath,DataPathComputeBinding
from azureml.pipeline.core import PipelineData,PipelineParameter,Pipeline
import json
from env_variables import ENV

def data_path(datastore,path_on_datastore):
    datapath = DataPath(datastore=datastore,
                        path_on_datastore=path_on_datastore)
    datapath_input = PipelineParameter('data_path',default_value=datapath)
    input_datapath = (datapath_input,DataPathComputeBinding(mode='mount'))
    return input_datapath

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
    
    cpu_compute_target = use_or_create_computetarget(ws=ws,
                                                     e=env,
                                                     cpu_cluster=True,
                                                     compute_name='score-cluster',
                                                     )
    gpu_compute_target = use_or_create_computetarget(ws=ws,
                                                     e=env,
                                                     cpu_cluster=False,
                                                     compute_name='gpu-cluster')
    datastore = use_or_create_datastore(ws=ws,
                                        datastore_name=env.datastore_name,
                                        container_name='news-clf',
                                        account_name=config['account_name'],
                                        account_key=config['account_key'],
                                        use_default=False)
    environment = use_or_create_environment(ws=ws,
                                            env_name='news_clf_dependencies',
                                            conda_dependencies='yaml/run_dependencies.yml',
                                            create_new_env=True)
    
    model_name_param = PipelineParameter('model_name',default_value='news_clf_model.pt')
    model_version_param = PipelineParameter('model_version',default_value='0')
    cuda_param = PipelineParameter('cuda',default_value='true')
    batch_train_param = PipelineParameter('batch_train',default_value=32)
    batch_eval_param = PipelineParameter('batch_eval',default_value=32) 
    tokenizer_param = PipelineParameter('tokenizer',default_value='treebank')
    vocab_param = PipelineParameter('vocab',default_value='vocab')
    is_sentence_param = PipelineParameter('is_sentence',default_value='false')
    max_seq_len_param = PipelineParameter('max_seq_len',default_value=256)
    train_corpus_param = PipelineParameter('train_corpus',default_value='train_corpus')
    eval_corpus_param = PipelineParameter('eval_corpus',default_value='eval_corpus')
    mode_param = PipelineParameter('mode',default_value='skipgram')
    hidden_size_param = PipelineParameter('hidden_size',default_value=128)
    num_layers_param = PipelineParameter('num_layers',default_value=2)
    dropout_param = PipelineParameter('dropout',default_value=0.5)
    embedding_size_param = PipelineParameter('embedding_size',default_value=100)
    embedding_trainable_param = PipelineParameter('embedding_trainable',default_value='true')
    use_word_embedding_param = PipelineParameter('use_word_embedding',default_value='true')
    bidirectional_param = PipelineParameter('bidirectional',default_value='true')
    learning_rate_param = PipelineParameter('learning_rate',default_value=0.01)
    epochs_param = PipelineParameter('epochs',default_value=50)
    batchscore_dataset_param = PipelineParameter('batchscore_dataset',default_value='batchscore_dataset.csv')
    output = PipelineData('output',datastore=datastore)
    
    runconfig = RunConfiguration()
    runconfig.environment = environment
    
    train_step = PythonScriptStep(name='train_step',
                                  source_directory='.',
                                  script_name='py/train.py',
                                  arguments=[
                                             '--model_name',model_name_param,
                                             '--cuda',cuda_param,
                                             '--batch_size_train',batch_train_param,
                                             '--batch_size_eval',batch_eval_param,
                                             '--tokenizer',tokenizer_param,
                                             '--vocab',vocab_param,
                                             '--is_sentence',is_sentence_param,
                                             '--max_seq_len',max_seq_len_param,
                                             '--train_corpus',train_corpus_param,
                                             '--eval_corpus',eval_corpus_param,
                                             '--mode',mode_param,
                                             '--hidden_size',hidden_size_param,
                                             '--num_layers',num_layers_param,
                                             '--dropout',dropout_param,
                                             '--embedding_size',embedding_size_param,
                                             '--embedding_trainable',embedding_trainable_param,
                                             '--use_word_embedding',use_word_embedding_param,
                                             '--bidirectional',bidirectional_param,
                                             '--learning_rate',learning_rate_param,
                                             '--epochs',epochs_param,
                                             '--save_path',output
                                             ],
                                  outputs=[output],
                                  compute_target=gpu_compute_target,
                                  runconfig=runconfig,
                                  allow_reuse=True
                                    )
    print('train_step built')
    
    register_step = PythonScriptStep(name='register_step',
                                     source_directory='.',
                                     script_name='azure/register.py',
                                     arguments=['--model_name',model_name_param,
                                                '--input',output],
                                     inputs=[output],
                                     compute_target=cpu_compute_target,
                                     runconfig=runconfig,
                                     allow_reuse=False)
    print('register_step built')
    register_step.run_after(train_step)
    
    batchscore_step = PythonScriptStep(name='batchscore_step',
                                       script_name='azure/batch_score.py',
                                       source_directory='.',
                                       arguments=['--model_name',model_name_param,
                                                  '--model_version',model_version_param,
                                                  '--dataset',batchscore_dataset_param,
                                                  '--cuda',cuda_param,
                                                  '--batch_size',batch_eval_param],
                                       compute_target=gpu_compute_target,
                                       runconfig=runconfig,
                                       allow_reuse=False)
    print('batchscore_step built')
    batchscore_step.run_after(register_step)
    
    pipeline = Pipeline(workspace=ws,
                        steps=[train_step,register_step,batchscore_step])
    pipeline.publish(name=env.pipeline_name,
                     version=env.build_id)
    

if __name__ == '__main__':
    main()
