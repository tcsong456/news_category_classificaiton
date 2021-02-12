import argparse
import sys
sys.path.append('azure')
sys.path.append('py')
from azure_utils import get_model,use_or_create_environment
from azureml.core.webservice import AksWebservice
from azureml.core.compute import AksCompute,ComputeTarget
from azureml.core.model import InferenceConfig,Model
from azureml.core.run import Run
from azureml.exceptions import ComputeTargetException

def parseargs():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_name',type=str)
    arg('--model_version',type=str)
    arg('--service_name',type=str)
    arg('--env_name',type=str)
    arg('--vm_size',type=str)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    run = Run.get_context()
    ws = run.experiment.workspace

    model = get_model(ws=ws,
                      model_name=args.model_name,
                      model_version=args.model_version)
    
    environment = use_or_create_environment(ws=ws,
                                            env_name=args.env_name)
    
    try:
        aks_compute = AksCompute(workspace=ws,
                                 name=args.service_name)
    except ComputeTargetException:
        aks_compute_config = AksCompute.provisioning_configuration(vm_size=args.vm_size)
        aks_compute = ComputeTarget.create(workspace=ws,
                                           name=args.service_name,
                                           provisioning_configuration=aks_compute_config)
        aks_compute.wait_for_completion(show_output=True)
    
    inference_config = InferenceConfig(entry_script='deployment/score.py',
                                       environment=environment,
                                       source_directory='.')
    aks_config = AksWebservice.deploy_configuration()
    aks_service = Model.deploy(workspace=ws,
                               name=args.service_name,
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
    n_rounds = np.ceil(len(input) / size).astype(int)
    with torch.no_grad():
        for i in range(n_rounds):
            if i < n_rounds - 1:
                pred = model(input[size*i:size*(i+1)])
            else:
                pred = model(input[size*i:])
            preds.append(np.argmax(pred.cpu().numpy(),axis=1))
