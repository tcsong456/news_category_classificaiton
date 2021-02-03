from azure_utils import *
from azureml.core.authentication import ServicePrincipalAuthentication
import json
from env_variables import ENV

def main():
    with open('config.json','r') as f:
        config = json.load(f)        
    auth = ServicePrincipalAuthentication(tenant_id=config['tenant_id'],
                                          service_principal_id=config['service_principal_id'],
                                          service_principal_password=config['service_principal_password'])
    
    ws = use_or_create_workspace(workspace=config['workspace'],
                                 resource_group=config['resource_group'],
                                 subscription_id=env.subscription_id,
                                 auth=auth) 
    
    
    

if __name__ == '__main__':
    