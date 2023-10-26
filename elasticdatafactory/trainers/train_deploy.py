# import functions for interacting with the operating system
import os
# import json to parse hyperparams
import json
# import argparse for parsing env. vars
import argparse
# import AWS SDK for Python
import boto3
# import pathlib for path extraction
from pathlib import Path
# import pandas
import pandas as pd
# pickle to serialize and save model output
import pickle as pkl
# sagemaker entry point
from sagemaker_containers import entry_point
# sagemaker distributed training
from sagemaker_xgboost_container import distributed
# import classifier frxgboost
import xgboost as xgb
# import logger to log output
import logging

# load logger
logger = logging.getLogger('')

# create session and set environment variables
session = boto3.Session()
s3 = boto3.resource('s3')
# provides APIs for creating and managing S3 resources
s3_client = boto3.client('s3')

def load_data(input_path: str, header=False) -> pd.DataFrame:
    
    try:
        files = [file.as_posix() for file in Path(input_path).iterdir() if file.is_file()]
        if header == False:
            df_final = pd.read_csv(files[0], header=None)
            if len(files) > 1:
                for file in files[1:]:
                    df = pd.read_csv(file, header=None)
                    df_final = pd.concat([df_final, df])
        else:
            df_final = pd.read_csv(files[0])
            if len(files) > 1:
                for file in files[1:]:
                    df = pd.read_csv(file)
                    df_final = pd.concat([df_final, df])
    except Exception as e:
        logging.error('Exception occurred', exc_info=True)
        raise ValueError('Training channel must have data to train model')
    try:
        y = df_final[0]
        X = df_final.drop(0, axis=1)
    except Exception as e:
        logging.error('Exception occurred', exc_info=True)
        raise ValueError('Improper specification of target column')
    
    return X, y

def train_model(is_master, train_input=None, validation_input=None, hyperparameters={}, model_data='.'):
    """Train model using XGBoost framework
    
    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training,
                        or is running single node training job.
                        Note that rabit_run includes this argument.
    """
    # load training data
    if train_input is not None:
        X, y = load_data(train_input)
        dtrain = xgb.DMatrix(X, y)
    else:
        dtrain = None
    
    # load validation data
    if validation_input is not None:
        X, y = load_data(validation_input)
        dval = xgb.DMatrix(X, y)
    else:
        dval = None

    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    # train xgboost model without checkpoints
    try:
        # replace num_round with num_boost_round
        num_boost_round = int(hyperparameters['num_round'])
        # remove num_round from params
        hyperparameters.pop('num_round', None)
        # train xgb model
        boosted_model = xgb.train(params=hyperparameters,
                                  dtrain=dtrain,
                                  evals=watchlist,
                                  num_boost_round=num_boost_round,
                                 )
    except Exception as e:
        logging.error('Exception occurred during model training', exc_info=True)

    # Save the model to the location specified by ``model_dir``
    if is_master:
        model_location = model_data + '/xgboost-model.pkl'
        pkl.dump(boosted_model, open(model_location, 'wb'))
        logging.info('Stored trained model at {}'.format(model_location))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument('--hyper_params', default=os.environ.get('SM_HPS'))

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_data', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train_input', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation_input', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--sm_hosts', type=str, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--sm_current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    args, unknown = parser.parse_known_args()
    
    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host
    # Get Hyperparameters for modle training
    hyperparameters = json.loads(args.hyper_params)
    
    training_args = dict(
        train_input=args.train_input,
        validation_input=args.validation_input,
        hyperparameters=hyperparameters,
        model_data=args.model_data
        )

    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=train_model,
            args=training_args,
            include_in_training=(args.train_input is not None),
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True
        )
    else:
        # If single node training, call training method directly.
        if args.train_input:
            training_args['is_master'] = True
            training_args(**training_args)
        else:
            raise ValueError('Training channel must have data to train model')