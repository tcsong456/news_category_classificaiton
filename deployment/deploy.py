import argparse
import sys
sys.path.append('azure')
from azure_utils import get_model
from azureml.core.webservice import AciWebService
from azureml.core.model import InferenceConfig,Model
from azureml.core.authentication import ServicePrincipalAuthentication

def parseargs():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_name',type=str)
    arg('--model_version',type=str)
    arg('--service_name',type=str)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    