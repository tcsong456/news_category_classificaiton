from azureml.core import Workspace,Environment,Datastore,Model
from azureml.exceptions import WorkspaceException,UserErrorException,ProjectSystemException
from azureml.core.compute import AmlCompute,ComputeTarget
from azureml.core.runconfig import DEFAULT_CPU_IMAGE,DEFAULT_GPU_IMAGE

def use_or_create_workspace(workspace_name,
                            resource_group,
                            subscription_id,
                            auth,
                            location='westeurope'):
    create_new = False
    try:
        ws = Workspace.get(name=workspace_name,
                           resource_group=resource_group,
                           subscription_id=subscription_id,
                           auth=auth)
    except WorkspaceException:
        print('wrong workspace name provided')
        create_new = True
    except UserErrorException:
        print('no access to the subscription provided,probably non-existing subscritpion')
        create_new = True
    except ProjectSystemException:
        print('wrong resource_group name provided')
        create_new = True
    
    if create_new:
        ws = Workspace.create(name=workspace_name,
                              resource_group=resource_group,
                              subscription_id=subscription_id,
                              location=location,
                              auth=auth)
    
    return ws

def use_or_create_computetarget(ws,
                                e,
                                cpu_cluster,
                                compute_name,
                                batch_scoring=False):
    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target is not None and type(compute_target) == AmlCompute:
            print(f'found compute target {compute_name}')
    else:
        if cpu_cluster:
            if batch_scoring:
                vm_size = e.cpu_vm_size_scoring
            else:
                vm_size = e.cpu_vm_size
        else:
            if batch_scoring:
                vm_size = e.gpu_vm_size_scoring
            else:
                vm_size = e.gpu_vm_size
        
        config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                       vm_priority=e.vm_priority_scoring if batch_scoring else e.vm_priority_scoring,
                                                       max_nodes=e.max_nodes_scoring if batch_scoring else e.max_nodes,
                                                       min_nodes=e.min_nodes_scoring if batch_scoring else e.min_nodes)
        compute_target = ComputeTarget.create(workspace=ws,
                                              name=compute_name,
                                              provisioning_configuration=config)
        compute_target.wait_for_completion(show_output=True)
    
    return compute_target

def use_or_create_environment(ws,
                              env_name,
                              conda_dependencies=None,
                              enable_docker=False,
                              use_gpu=False,
                              create_new_env=False,
                              overwrite=False):
    assert env_name is not None,'env name must be provided'
    env_list = Environment.list(ws)
    if overwrite:
        assert env_name in env_list,'when overwrite is activated,there must be an existing env to be overwritten'
    if env_name in env_list and not overwrite:
        environment = env_list[env_name]
        return environment
    elif create_new_env or (overwrite and conda_dependencies is not None):
        environment = Environment.from_conda_specification(name=env_name,
                                                           file_path=conda_dependencies)
    else:
        raise ValueError('you can either create or use a environment')
    
    if enable_docker:
        if use_gpu:
            environment.docker.base_image = DEFAULT_GPU_IMAGE
        else:
            environment.docker.base_iamge = DEFAULT_CPU_IMAGE
    
    environment.register(ws)
    
    return environment

def use_or_create_datastore(ws,
                            datastore_name,
                            container_name=None,
                            account_name=None,
                            account_key=None,
                            use_default=True):
    if use_default:
        datastore = ws.get_default_datastore()
        return datastore

    datastores = ws.datastores
    if datastore_name in datastores:
        datastore = datastores[datastore_name]
    else:
        assert container_name is not None,'when registering a blob container,container name must be given'
        assert account_name is not None,'when registering a blob container,account name must be given'
        assert account_key is not None,'when registering a blob container,account key must be given'
        datastore = Datastore.register_azure_blob_container(workspace=ws,
                                                            datastore_name=datastore_name,
                                                            container_name=container_name,
                                                            account_name=account_name,
                                                            account_key=account_key)
    
    return datastore

def get_model(ws,
              model_name,
              model_version=None,
              tags=None):
    model_version = int(model_version)
    if model_version >= 0:
        print('loading from model')
        model = Model(workspace=ws,
                      name=model_name,
                      version=model_version,
                      tags=tags)
    else:
        print('loading from model list')
        models = Model.list(workspace=ws,
                            name=model_name,
                            latest=True)
        model = models[-1]
    
    if model is None:
        raise FileNotFoundError('model not found')
    
    return model
    
    
        
        
        #%%
