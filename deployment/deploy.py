import sys
sys.path.append('azure')
import json
from env_variables import ENV
from azure_utils import get_model,use_or_create_environment,use_or_create_workspace
from azureml.core.webservice import AksWebservice
from azureml.core.compute import AksCompute,ComputeTarget
from azureml.core.model import InferenceConfig,Model
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.exceptions import ComputeTargetException

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
    
    model = get_model(ws=ws,
                      model_name=env.model_name,
                      model_version=env.model_version)
    
    environment = use_or_create_environment(ws=ws,
                                            env_name=env.experiment_name)
    
    try:
        aks_compute = AksCompute(workspace=ws,
                                 name=env.aks_service_name)
    except ComputeTargetException:
        aks_compute_config = AksCompute.provisioning_configuration(
                                                                    vm_size=env.scoring_vm_size
                                                                    )
        aks_compute = ComputeTarget.create(workspace=ws,
                                           name=env.AKS_SERVICE_NAME,
                                           provisioning_configuration=aks_compute_config)
        aks_compute.wait_for_completion(show_output=True)
        
    inference_config = InferenceConfig(entry_script='deployment/score.py',
                                       environment=environment,
                                       source_directory='.')
    aks_config = AksWebservice.deploy_configuration()
    aks_service = Model.deploy(workspace=ws,
                               name=env.AKS_SERVICE_NAME,
                               models=[model],
                               inference_config=inference_config,
                               deployment_config=aks_config,
                               deployment_target=aks_compute,
                               overwrite=True)
    aks_service.wait_for_deployment(show_output=True)    
    print(aks_service.state)

if __name__ == '__main__':
    main()
    
#%%
#from azureml.core import Workspace
#from azureml.core.webservice import AksWebservice
#from azureml.core.compute import AksCompute
#from azureml.exceptions import ComputeTargetException
##ws = Workspace.get(name='aml-workspace',
##                   resource_group='aml-resource-group',
##                   subscription_id='64c727c2-4f98-4ef1-a45f-09eb33c1bd59')
##aks_config = AksWebservice(ws,name=
#try:
#    aks_compute = AksCompute(workspace=ws,
#                             name='newsclfaciservice')
#except ComputeTargetException:
#    print('wrong!')
##
#%%
#try:
#    aks_compute = AksCompute(workspace=ws,
#                             name='newsclfaksservice')
#except ComputeTargetException:
#    aks_compute_config = AksCompute.provisioning_configuration(vm_size='STANDARD_NC6')
#    aks_compute = ComputeTarget.create(workspace=ws,
#                                       name='abc',
#                                       provisioning_configuration=aks_compute_config)
#    aks_compute.wait_for_completion(show_output=True)
