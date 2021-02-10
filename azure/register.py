from azureml.core.model import Model
from azureml.core.run import Run
import torch
import argparse
import json
import os
import sys
sys.path.append('py')

def register_model(run_id,
                   exp,
                   model_path,
                   model_name,
                   mtags=None):
    try:
        tags = {'title':'news_classification',
                'run_id':run_id,
                'exp_name':exp.name}
        if mtags is not None:
            tags.update(mtags)
        
        Model.register(workspace=exp.workspace,
                       model_path=model_path,
                       model_name=model_name,
                       tags=tags)
        print(f'model:{model_name} has been sucessfully registered')
    except Exception as error:
        print(error)
        exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str)
    parser.add_argument('--input',type=str)
    args = parser.parse_args()
    
    run = Run.get_context()
    exp = run.experiment
    
    with open('config.json','r') as f:
        config = json.load(f)
    register_tags = config['registration']
    mtags = {}
    print(run.parent.get_metrics())
    for tag in register_tags['tags']:
        try:
            if tag == 'eval_avg_acc':
                value = max(run.parent.get_metrics()[tag])
            else:
                value = run.parent.get_metrics()[tag]
            mtags[tag] = value
        except KeyError:
            print(f'{tag} key not found')
    
    model_path = os.path.join(args.input,args.model_name)
    model = torch.load(model_path,map_location=torch.device('cpu'))
    
    if model is not None:
        run_id = run.id
        register_model(run_id=run_id,
                       exp=exp,
                       model_path=model_path,
                       model_name=args.model_name,
                       mtags=mtags)
    else:
        raise Exception('model is not found')

if __name__ == '__main__':
    main()
#%%
