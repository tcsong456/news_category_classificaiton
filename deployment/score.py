from azureml.core import Model
import numpy as np
import os
import torch
import json

def init():
    global model
    model_path = os.environ.get('AZUREML_MODEL_DIR')
    model_path = Model.get_model_path(model_path.split('/')[-2])
    model = torch.load(model_path)

def run(data):
    raw_data = json.loads(data)['data']
    raw_data = torch.tensor(raw_data,dtype=torch.long)
    batch_size = 128
    n_rounds = np.ceil(len(raw_data) / batch_size).astype(int)
    output = None
    for i in range(n_rounds):
        if i < n_rounds - 1:
            data = raw_data[i*batch_size:(i+1)*batch_size]
        else:
            data = raw_data[i*batch_size:]
        data = data.cuda() if torch.cuda.is_available() else data
        pred = model(data)
        pred = pred.cpu().data.numpy()
        output = pred if output is None else np.vstack([output,pred])
    output = [row.tolist() for row in output]
        
    return output