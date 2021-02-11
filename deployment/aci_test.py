import argparse
import json
import pandas as pd
from azureml.core.webservice import AciWebservice
from azureml.core.run import Run
from azureml.core import Dataset


def main():
    parser = argparse.ArgumentError()
    arg = parser.argument_name
    arg('--service_name',type=str)
    arg('--datastore_name',type=str)
    arg('--train_corpus',type=str)
    arg('--eval_corpus',type=str)
    arg('--frac',type=float)
    args = parser.parse_args()
    
    run = Run.get_context()
    ws = run.experiment.workspace
        
    datastore = ws.datastores[args.datastore_name]  
    train_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,args.train_corpus)).to_pandas_dataframe()
    eval_corpus = Dataset.Tabular.from_delimited_files(path=(datastore,args.eval_corpus)).to_pandas_dataframe()
    corpus = pd.concat([train_corpus,eval_corpus]).sample(frac=args.frac).reset_index(drop=True)
    
    df = []
    for _,row in corpus.iterrows():
        df.append([row['label'],row['text']])
    json_data = json.dumps({'data':df})
    json_data = bytes(json_data,encoding='utf-8')
    
    aci_service = AciWebservice(workspace=ws,
                                name=args.service_name)
    
    try:
        acc,loss,output = aci_service.run(json_data)
        output.to_csv('score.csv',index=False)
        datastore.upload_files(files=['score.csv'],
                               target_path='score',
                               overwrite=True)
        print('successfully saved score result')
    except Exception as error:
        print(error)

if __name__ == '__main__':
    main()

#%%


    