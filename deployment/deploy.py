import argparse
import sys
sys.path.append('azure')
from azure_utils import get_model,use_or_create_environment
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig,Model
from azureml.core.run import Run

def parseargs():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_name',type=str)
    arg('--model_version',type=str)
    arg('--service_name',type=str)
    arg('--env_name',type=str)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    run = Run.get_context()
    ws = run.experiment.workspace
    aci_config = AciWebservice.deploy_configuration()
    model = get_model(ws=ws,
                      model_name=args.model_name,
                      model_version=args.model_version)
    environment = use_or_create_environment(ws=ws,
                                            env_name=args.env_name)
    
    inference_config = InferenceConfig(entry_scrpt='deployment/score.py',
                                       environment=environment,
                                       source_directory='.')
    aci_service = Model.deploy(workspace=ws,
                               name=args.service_name,
                               models=[model],
                               inference_config=inference_config,
                               deployment_config=aci_config,
                               overwrite=True)
    aci_service.wait_for_deployment(show_output=True)    
    print(aci_service.state)

if __name__ == '__main__':
    main()
    
#%%
