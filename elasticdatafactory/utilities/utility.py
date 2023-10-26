# import necessary libraries
# import functions for interacting with the operating system
import os
# import capability to “pretty-print” arbitrary Python data structures
from pprint import pprint
# import glob to retrieve files/pathnames matching a specified pattern
from glob import glob
# importlib to import objects by label
import importlib
# import to populate python objects
import inspect
# import urlib to parse url
from urllib.parse import urlparse
# import StringIO to read and write a string buffer
from io import StringIO, BytesIO
# json for data conversion and storage
import json
# import pickle for serializing data
import pickle as pkl
# import tarfile to read and write tar archives including gzip, bz2 and lzma compression
import tarfile
# import sagemaker
import sagemaker
# import awswrangler for data wrangling
import awswrangler as wr
# import pandas for relational data analysis and manipulation
import pandas as pd
# import numpy for calculation with arrays
import numpy as np
# data partitioning for model training and validation
from sklearn.model_selection import (
    TimeSeriesSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    GroupShuffleSplit,
    GroupKFold,
    StratifiedShuffleSplit,
    StratifiedGroupKFold,
)
# import stats for chi-square computation
from scipy import stats
# import mlflow for trial tracking
import mlflow
# model staging and tracking
from mlflow.tracking import MlflowClient
# import mlflow_connect
from mlflow_connect.mlflow_connect import MLflowConnect
# import sagemaker model logging
from mlflow_connect.config import get_subnets, get_security_groups
# log models with a signature that defines the schema of the model's inputs and outputs
from mlflow.models.signature import infer_signature
# import mlflow environment definitions
from mlflow.utils.environment import _mlflow_conda_env
# import re for regex match and search
import re
# import hashlib for hashing of a file/file-like object
import hashlib
# import ast to evaluate the string as python expression 
import ast
# import sql translator
import sqlglot
# implementation of tree data structure
from treelib import Node, Tree
# import progress bar
from tqdm.auto import tqdm
# import logger for logs
import logging
# import classifier xgboost
import xgboost as xgb
# import plot data class for data visualization
from elasticdatafactory.plots import plotter
# mlflow client
mlflow_client = MlflowClient()
# create new `pandas` methods which use `tqdm` progress
tqdm.pandas()
# create mlflow connect method
MLflowConnect()
# load logger
logger = logging.getLogger('')

"""
This module provides utility functions that are used within Data Factory, 
and these can be useful for external consumption as well.
"""

def create_tree(path, parent=None, tree=None, display=True):
    
    """
    Provides an overview of project directories and files 
    in the form of a tree data structure

    Parameters
    ----------
    path : str
        A path to project folder containing all directories, sub-directories, and files
    parent : str
        Represents name of a parent node for a child node, by default it is set to None
    tree : Tree()
        A Tree object created through Treelib opensource library, by default it is None
        Source: https://treelib.readthedocs.io/en/latest/
    display : bool
        Displays Tree Data Structure when set to True
    
    Returns
    -------
    tree : Tree()
        A Tree object created through Treelib opensource library
        
    Examples
    --------
    # Create a tree object
    create_tree(path='root/path_to_project/')
    # Optional arguments
    create_tree(path='root/path_to_project/', parent='parent-node', tree=Tree())
    
    """

    if parent is None:
        parent = os.path.basename(os.path.dirname(path)).lower()
    if tree is None:
        # create tree object
        tree = Tree()
        # create the root node
        tree.create_node(os.path.basename(os.path.dirname(path)).title(), os.path.basename(os.path.dirname(path)).lower())
    elif tree.root is None:
        # create the root node
        tree.create_node(os.path.basename(os.path.dirname(path)).title(), os.path.basename(os.path.dirname(path)).lower())
    
    if os.path.isdir(path) == True:
        # populate all directories and sub-directories
        dir_names = [os.path.basename(os.path.dirname(dir)) for dir in glob(path + '[!_]*/', recursive=True) if os.path.isdir(dir)]
        for directory in dir_names:
            tree.create_node(directory, directory.lower(), parent=parent)
        # populate files within each directory
        file_names = [os.path.basename(dir) for dir in glob(path + '[!_]*', recursive=True) if not os.path.isdir(dir)]
        for file in file_names:
            tree.create_node(file, parent=parent)
        if display is True:
            # display tree data structure
            tree.show()
    
    return tree

def extract_values_from_dict(input_dict):

    """
    Given a dictionary, return the value of the key value pairs in the input dictionary

    Parameters:
    -----------
    input_dict : dictionary
        Dictionary with key value pairs

    Returns:
    --------
    value_list : list
        List of values in the key value paris of the dictionary
    """
    
    value_list = []
    try:
        for key, val in input_dict.items():
            value_list.extend([element for element in (val or [])])
        return value_list
    except Exception as e:
        print('[INFO] failed to extract values from dictionary %s', e)           

def import_json(file_name):
    
    """
    Import JSON file from a specific path

    Parameters:
    -----------
    file_name : str
        Path for the json file

    Returns:
    --------
    data : dictionary
        Imported json file
    """
    
    try:
        with open(file_name, 'r') as input_file:
            data = json.loads(input_file.read())
            return data
    except Exception as e:
        print('[INFO] JSON failed to load, an exception occurred while processing the request %s', e)
        
def export_json(file_name, data):
    
    """
    Export JSON file to a specific path

    Parameters:
    -----------
    file_name : str
        Path for the json file that will be exported
    data : dictionary
        Data to be exported to the path

    Returns:
    --------
    None
    
    """
    
    try:
        with open(file_name, "w") as output_file:
            json.dump(data, output_file)
            return '[INFO] JSON export successful'
    except TypeError:
        print('[INFO] JSON export failed')
        
def populate_modules(module_name='query_registry'):
    
    """
    Return members of a module in a dictionary of (name, value) pairs sorted by name
    if the member is a class object
    and if the member's __module__ attribute equals the loaded moduel's __name__ attribute

    Parameters:
    -----------
    module_name : string, default='query_registry'
        name of the module

    Returns:
    --------
    object_dict : dictionary
        dictionary of the member name and member
    
    """
    
    object_dict = dict()
    loaded_module = importlib.import_module(module_name)
    for object_name, obj in inspect.getmembers(loaded_module, lambda x: inspect.isclass(x) and 
                                               x.__module__ == loaded_module.__name__):
        try:
            if inspect.isclass(obj):
                object_dict[object_name] = obj
        except ValueError:
            continue
    return object_dict

def load_json_from_s3(file_name, bucket_name='sagemaker-us-east-2-534295958235', prefix='rcd/', s3_client=None):
    
    """
    Load JSON file from Amazon Web Services (AWS) S3 bucket

    Parameters:
    -----------
    file_name : str
        name of the json file
    bucket_name : str
        s3 bucket name. Default 'sagemaker-us-east-2-534295958235'
    prefix : str
        path of the json file of interest. Default 'rcd/queryregistry_unittest/'
    s3_client : botocore.client.S3
        boto3 client name. E.g., boto3.client('s3')

    Returns:
    --------
    json_content : dictionary
        json content of the target location
    """
    
    try:
        result = s3_client.get_object(Bucket=bucket_name, Key=prefix+file_name)
        json_content = json.loads(result["Body"].read().decode('utf-8'))
        return json_content
    except Exception as e:
        print('[INFO] failed to load json from S3 bucket %s', e)
        
def save_json_to_s3(data_dict, file_name, bucket_name='sagemaker-us-east-2-534295958235', prefix='rcd/', s3_client=None):
    
    """
    Load JSON file from Amazon Web Services (AWS) S3 bucket

    Parameters:
    -----------
    data_dict : dict()
        name of data dictionary to export as json file
    file_name : str
        name of the json file
    bucket_name : str
        s3 bucket name. Default 'sagemaker-us-east-2-534295958235'
    prefix : str
        path of the json file of interest. Default 'rcd/queryregistry_unittest/'
    s3_client : botocore.client.S3
        boto3 client name. E.g., boto3.client('s3')

    Returns:
    --------
        None
    """
    
    try:
        # Convert Dictionary to JSON String
        json_content = json.dumps(data_dict, indent=2, default=str)
        s3_client.put_object(Bucket=bucket_name, Key=prefix+file_name, Body=json_content)
        print('[INFO] data exported successfully!')
    except Exception as e:
        print('[INFO] failed to load json from S3 bucket %s', e)
        
class ImportExportCSV():
    
    """
    Exports raw user data to an Amazon Web Services (AWS) S3 bucket with CSV uploads.
    
    Attributes
    ----------
    data_frame : Pandas DataFrame
        A DataFrame containing data to be imported or exported as comma-separated values
    bucket : str
        A string containing the name of Amazon Web Services (AWS) S3 bucket
    prefix : str
        A string containing path to the directory where files are either exported to or imported from
    file_name : str
        A string containing the name of the file to be imported or exported in .csv format

        
    Examples
    --------
    # Export data to S3 bucket given path to export directory and file name
    bucket = 'sagemaker-us-east-2-534295958235'
    prefix = 'rm283'
    file_name = 'adobe_dq_prcd_clickstream_data_parsed'
    export_object = ImportExportCSV(data_frame, bucket, prefix, file_name)
    export_object.export_csv(index=True)
    # Import data from S3 bucket given path to import directory and file name 
    import_object = ImportExportCSV(pd.DataFrame(), bucket, prefix, file_name)
    data_frame = import_object.import_csv()
        
    """
    
    def __init__(self, data_frame, bucket, prefix, file_name):
        
        self.data_frame = data_frame
        self.bucket = bucket
        self.prefix = prefix
        self.file_name = file_name + '.csv'
        
    
    def export_csv(self, index=False):
    
        """
        Save dataframe to a .csv file in Amazon Web Services (AWS) S3 bucket
        
        Parameters
        ----------
        self : object
            ImportExportCSV class instance.
        index : boolean, default=False
            A data object representing row names (index).
    
        Returns
        -------
        status : boolean
            A data object representing status of export.
        """

        # Write dataframe to s3
        path = f's3://{self.bucket}/{self.prefix}/{self.file_name}'
        # incremental upload to an already created bucket
        self.data_frame.to_csv(path, index=index)
    
    def import_csv(self, converters={}):

        """
        Import dataframe from a .csv file in Amazon Web Services (AWS) S3 bucket
        
        Parameters
        ----------
        self : object
            ImportExportCSV class instance.
        converters : dict, default={}
            A dictionary to specify literal eval. to parse text enclosed in brackets
            
        Returns
        -------
        data_frame : Pandas DataFrame
            A DataFrame containing imported data from a .csv file
        """
        
        # path to .csv file
        path = f's3://{self.bucket}/{self.prefix}/{self.file_name}'
        # import .csv file as dataframe
        self.data_frame = wr.s3.read_csv(path=path, converters=converters)

        return self.data_frame
    
def read_parquet_s3(bucket, key, s3_client=None, **args):
    
    """
    Read a single Parquet file from provided Amazon Web Services (AWS) S3 bucket

    Parameters:
    -----------
    bucket : str
        S3 Bucket name containing objects
    key : str
        The name that is assigned to an object to retrieve 
    s3_client : botocore.client.S3
        boto3 client name. E.g., boto3.client('s3')

    Returns:
    --------
    dataset : Pandas DataFrame
        Dataset imported from AWS S3 bucket and converted to Pandas DataFrame
    """
    
    if s3_client is None:
        logger.error(f'No botocore.client.S3 provided!')
        return pd.DataFrame()
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    dataset = pd.read_parquet(BytesIO(obj['Body'].read()), **args)
    return dataset

def read_multiple_parquets_s3(bucket, prefix, s3=None, s3_client=None, verbose=False, **args):

    """
    Read multiple Parquet files from provided Amazon Web Services (AWS) S3 bucket

    Parameters:
    -----------
    bucket : str
        S3 Bucket name containing objects
    prefix : str
        Prefix of the object to retrieve from S3 bucket
    s3 : botocore.resource.S3
        S3 service resources. E.g., boto3.resource('s3')
    s3_client : botocore.client.S3
        boto3 client name. E.g., boto3.client('s3')
    verbose : bool
        A boolean indicating 

    Returns:
    --------
    dataset : Pandas DataFrame
        Dataset imported from AWS S3 bucket and converted to Pandas DataFrame
    """

    if not prefix.endswith('/'):
        prefix = prefix + '/'
    if s3_client is None:
        logger.error(f'No botocore.client.S3 provided!')
        return pd.DataFrame()
    if s3 is None:
        logger.error(f'No botocore.resource.S3 provided!')
        return pd.DataFrame()
    s3_keys = [item.key for item in s3.Bucket(bucket).objects.filter(Prefix=prefix)
               if item.key.endswith('.parquet')]
    if not s3_keys:
        logger.error(f'No parquet found in bucket {bucket} and prefix {prefix}')
        return pd.DataFrame()
    elif verbose:
        for p in s3_keys: 
            logger.info(f'Attempting to load parquet file: {p}')
    dfs = [read_parquet_s3(bucket=bucket, key=key, s3_client=s3_client, **args) 
           for key in s3_keys]
    return pd.concat(dfs, ignore_index=True)
    
def list_s3_objects(s3_uri, output_data_uri='', sagemaker_session=None):
    
    """
    Function to list file in s3 bucket
    
    Parameters
    ----------
    s3_uri : str
        An S3 Unique Resource Identifier (URI) within the context of the S3 protocol
    output_data_uri : str
        An S3 Unique Resource Identifier (URI) within the context of the S3 protocol for output data
        By default, it is set to an empty string
    sagemaker_session : object
        Session object that manages interactions with SageMaker API operations and other AWS service
        E.g., sagemaker.Session()
    
    Returns
    -------
    objects : dict(bucket: list(prefix))
        returns a dictionary with keys representing S3 bucket, and values representing list of all objects
        in the prefix extracted from s3_uri
        
    """
    
    objects = {}
    if sagemaker_session is None:
        logger.error(f'No Sagemaker Session provided!')
        return objects
    object_paths = sagemaker.s3.S3Downloader.list(s3_uri=s3_uri, sagemaker_session=sagemaker_session)
    # enumerate file paths
    for path in object_paths:
        key = urlparse(s3_uri).netloc
        value = output_data_uri + '/' + os.path.basename(path)
        objects.setdefault(key,[]).append(value)
    return objects
    
def remove_s3(bucket, prefix):
    
    """
    Deletes all objects under the received prefix

    Parameters
    ----------
    bucket : str
        Bucket name containing objects
    prefix : str
        prefix identifying location of objects in s3 bucket
    
    Returns
    -------
        None
    """
    
    try:
        wr.s3.delete_objects(f's3://{bucket}/{prefix}')
        print('[INFO] objects removed successfully!')
    except Exception:
        print('[ERROR] failed to remove objects removed! Access Denied...')
        
def restore_s3(bucket, prefix, s3_client):
    
    """
    Restores all previously deleted objects under the received prefix
    
    Parameters
    ----------
    bucket : str
        Bucket name containing objects
    prefix : str
        prefix identifying location of objects in s3 bucket
    s3_client : botocore.client.S3
        boto3 client name. E.g., boto3.client('s3')

    Returns
    -------
        None
    """
    
    resp = s3_client.list_object_versions(Bucket=bucket, Prefix=prefix)
    latest = []
    print('[INFO] populating deleted objects...')
    for v in tqdm(resp['DeleteMarkers']):
        if v['IsLatest']:
            latest.append(v)
    truncated = resp['IsTruncated']
    print('[INFO] populating versions of deleted objects...')
    while(truncated):
        resp = s3_client.list_object_versions(Bucket=bucket,
                                              Prefix=prefix,
                                              KeyMarker=resp['NextKeyMarker'],
                                              VersionIdMarker=resp['NextVersionIdMarker']
                                             )
        for v in resp['DeleteMarkers']:
            if v['IsLatest']:
                latest.append(v)
        truncated = resp['IsTruncated']
    print('[INFO] deleting existing versions 0-byte to make previous latest...')
    for v in tqdm(latest):
        resp = s3_client.delete_object(Bucket=bucket, Key=v['Key'], VersionId=v['VersionId'])
    print('[INFO] process completed, data is restored!')
    
def normalize_and_hash_email_address(email_address):
    """Returns the result of normalizing and hashing an email address.

    For this use case, Google Ads requires removal of any '.' characters
    preceding "gmail.com" or "googlemail.com"

    Args:
        email_address: An email address to normalize.

    Returns:
        A normalized (lowercase, removed whitespace) and SHA-265 hashed string.
    """
    normalized_email = email_address.lower()
    email_parts = normalized_email.split("@")
    # Checks whether the domain of the email address is either "gmail.com"
    # or "googlemail.com". If this regex does not match then this statement
    # will evaluate to None.
    is_gmail = re.match(r"^(gmail|googlemail)\.com$", email_parts[1])

    # Check that there are at least two segments and the second segment
    # matches the above regex expression validating the email domain name.
    if len(email_parts) > 1 and is_gmail:
        # Removes any '.' characters from the portion of the email address
        # before the domain if the domain is gmail.com or googlemail.com.
        email_parts[0] = email_parts[0].replace(".", "")
        normalized_email = "@".join(email_parts)

    return normalize_and_hash(normalized_email)

def normalize_and_hash(s):
    """Normalizes and hashes a string with SHA-256.

    Private customer data must be hashed during upload, as described at:
    https://support.google.com/google-ads/answer/7474263

    Args:
        s: The string to perform this operation on.

    Returns:
        A normalized (lowercase, removed whitespace) and SHA-256 hashed string.
    """
    return hashlib.sha256(s.strip().lower().encode()).hexdigest()
        
def convert_data_type(dataset, col_name, curr_type, new_type, verbose=False):
    
    """
    Converts data type to the specified type in Pandas DataFrame
    
    Parameters
    ----------
    dataset : Pandas DataFrame
        Dataset to replace data type
    col_name : str
        A str representing column name to apply condition
    curr_type : str
        Default type to be replaced
    new_type : str
        A pre-defined type to replace curr_type data with
        
    Returns
    -------
    dataset : Pandas DataFrame
        Dataset with replaced types for col_name
    """
    
    # check if conversion is required
    if dataset[col_name].dtypes == new_type:
        if verbose:
            print(f'[INFO] "{col_name}" data type "{dataset[col_name].dtypes}" already in the specified format, process skipped...')
        return dataset
    # convert data type to the specified type if it is str
    elif dataset[col_name].dtypes == curr_type and new_type == 'str':
        if curr_type == 'float16' or curr_type == 'float32' or curr_type == 'float64':
            dataset[col_name] = dataset[col_name].round().astype('Int64')
        dataset[col_name] = dataset[col_name].astype(new_type)
    # convert data type to the specified type
    elif dataset[col_name].dtypes == curr_type:
        dataset[col_name] = dataset[col_name].astype(new_type)
    else:
        if verbose:
            print(f'[INFO] "{col_name}" current data type "{dataset[col_name].dtypes}" specified is inconsistent with '\
                  'provided data type, process skipped...')
    return dataset

def replace_values(dataset, col_name_condition, curr_val, new_val, col_name_replace=None):
    
    """
    Replace default value with a given value 
    
    Parameters
    ----------
    dataset : Pandas DataFrame
        Dataset to replace values from
    col_name_condition : str
        A str representing column name to apply condition
    col_name_replace : str
        A str representing column name to replace values
    curr_val : scalar value
        Default value to be replaced
    new_val : scalar value
        A pre-defined value to replace curr_val values with
        
    Returns
    -------
    dataset : Pandas DataFrame
        Dataset with replaced values
    """
    
    if col_name_replace is None:
        col_name_replace = col_name_condition
    dataset.loc[dataset[col_name_condition] == curr_val, col_name_replace] = new_val
    
    return dataset

def replace_nan_with_value(x, y, value):
    
    """
    Replace NaN values with a given value 
    
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    value : scalar value
        A pre-defined value to replace NaN values with
        
    Returns
    -------
    x : NumPy ndarray
        A sequence of categorical measurements
    y : NumPy ndarray
        A sequence of categorical measurements
    """
    
    x = np.array(
        [v if v == v and v is not None else value for v in x]
    )  # NaN != NaN
    y = np.array([v if v == v and v is not None else value for v in y])
    return x, y

def remove_incomplete_samples(x, y):
    
    """
    Drop NaN values in a sequence of measurements 
    
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
        
    Returns
    -------
    x : NumPy ndarray
        A sequence of categorical measurements
    y : NumPy ndarray
        A sequence of categorical measurements
    """
    
    x = [v if v is not None else np.nan for v in x]
    y = [v if v is not None else np.nan for v in y]
    arr = np.array([x, y]).transpose()
    arr = arr[~pd.isnull(arr).any(axis=1)].transpose()
    return arr[0], arr[1]
        
def convert_data_structure(data, to, copy=True):
    
    """
    Group row records by a specified column and collect row indices into a list within same group.
    
    Parameters
    ----------
    data : list / NumPy ndarray / Pandas DataFrame
        Input data to convert
    to : str
        Type specified for data conversion
    copy : boolean
        Specifies the option to manipulate a copy of the input data
    
    Returns
    -------
    converted : list / NumPy ndarray / Pandas DataFrame
        Output converted data in the specified type
    """
    
    converted = None
    if to == 'array':
        if isinstance(data, np.ndarray):
            converted = data.copy() if copy else data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.values
    elif to == 'list':
        if isinstance(data, list):
            converted = data.copy() if copy else data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == 'DataFrame':
        if isinstance(data, pd.DataFrame):
            converted = data.copy(deep=True) if copy else data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        logging.error(f'[ERROR] unknown data structure conversion: {to}')
    if converted is None:
        logging.error(f'[ERROR] cannot handle data structure conversion {type(data)} to {to}, ' \
                      'supported types are: array, list, DataFrame')
    else:
        return converted
    
def identify_columns_by_type(dataset, include):
    
    """
    Given a dataset, identify columns of the types requested.
    
    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame
    include : list of strings
        Desired column types
    
    Returns:
    --------
    A list of columns names
    
    Example:
    --------
    >>> df = pd.DataFrame({'col1': ['a', 'b', 'c', 'a'], 'col2': [3, 4, 2, 1], 'col3': [1., 2., 3., 4.]})
    >>> identify_columns_by_type(df, include=['int64', 'float64'])
    >>> ['col2', 'col3']
    """
    
    dataset = convert_data_structure(dataset, 'DataFrame')
    columns = list(dataset.select_dtypes(include=include).columns)
    return columns

def duplicate_keys(dataFrame, cols):
    
    """This is used to identify and count duplicate keys, i.e., any primary key with  
    duplicate entries is counted as one.
    
    Parameters
    ----------
    dataFrame : Pandas DataFrame
        data stored in dataFrame
    cols: list
        column names stored in a list

    Returns
    -------
    keys_with_duplicates : dict()
        Return keys with duplicate entries

    Examples
    --------
    >>>df = pd.DataFrame({'col1': ['a', 'a', 'c', 'd'], 'col2': [3, 4, 2, 1], 'col3': [1., 2., 3., 4.], 'col4': ['a', 'a', 'b', 'b']})
    >>>duplicate_keys(df, cols=['col1', 'col4'])
    {'col1': ['a'], 'col4': ['a']}
    """
    
    keys_with_duplicates = {}
    dataFrame = dataFrame.groupby(cols, as_index=False).size()
    for col in cols:
        keys_with_duplicates[col] = dataFrame[dataFrame['size'] > 1][col].to_list()
    
    return keys_with_duplicates

def data_profile(dataFrame):
    
    """Profile data to identify quality issues 
    
    Parameters
    ----------
    dataFrame : Pandas DataFrame
        data stored in dataFrame 

    Returns
    -------
    count_df : Pandas DataFrame
        Return data profile to show missingness rate 
    """
    
    # check for valid entries
    logger.info('[INFO] finding valid entries...')
    #check_valid = dataFrame.progress_apply(lambda x: sum(x.notnull()), axis=0).values # counts blank space as valid
    check_valid = dataFrame.progress_applymap(lambda x: np.nan if isinstance(x, str) and (not x or x.isspace()) else x).count().values
    # check for unique entries
    logger.info('[INFO] finding unique entries...')
    check_unique = dataFrame.nunique().values
    # check for duplicate keys
    logger.info('[INFO] finding duplicate keys...')
    check_duplicates = list()
    for col in tqdm(dataFrame.columns):
        keys = duplicate_keys(dataFrame, cols=[col])
        check_duplicates.append(len(keys[col]))
    # check for nulls
    logger.info('[INFO] finding NaN entries...')
    check_null = dataFrame.progress_apply(lambda x: sum(x.isna()), axis=0).values
    # get min and max values
    logger.info('[INFO] finding minimum and maximum entries...')
    categorical_columns = identify_columns_by_type(dataFrame, include=['object', 'category'])
    check_min = list()
    check_max = list()
    for col in tqdm(dataFrame.columns):
        if col not in categorical_columns:
            check_min.append(dataFrame[col].min())
            check_max.append(dataFrame[col].max())
        else:
            check_min.append('')
            check_max.append('')
    logger.info('[INFO] inspecting data types...')
    check_dtype = dataFrame.progress_apply(lambda x: str(x.dtypes), axis=0).values
    count_df = pd.DataFrame(data = list(zip(dataFrame.columns.tolist(), check_valid, check_unique, 
                                            check_duplicates, check_null, check_min, check_max, check_dtype)),
                            columns = ['Feature', 'Entries', 'Unique', 'Duplicate Keys', 'Null', 'Minimum', 'Maximum', 'Data Type']
                           )
    # highlight 'Feature' column 
    count_df = display(
        count_df, 
        rows=count_df.shape[0], 
        color='green', 
        columns_to_shadow=['Feature'], 
        columns_to_show=[])
    
    return count_df


def remove_outliers(dataset, cat_freq=100, std_threshold=3, lb_percentile=0.05, 
                    ub_percentile=0.95, lb_quantile=0.25, ub_quantile=0.75, num_method='IQD', exclude=[]):
    
    """
    Removing outliers from the dataset using the following logic:
    * if categorical variable, outliers are removed based on distribution of values using
        frequency as a cut-off threshold
    * if numerical variable, the following three methods are implemented:
        Mean and Standard Deviation Method (SD) - The mean and standard deviation of the residuals are calculated and compared. 
            If a value is a certain number of standard deviations away from the mean, that data point is identified as an outlier. 
            The specified number of standard deviations is called the threshold, and the default value is 3. 
            This method can fail to detect outliers since outliers increase the standard deviation. 
            The more extreme the outlier, the more the standard deviation is affected.
        
        Interquartile Deviation Method (IQD) - The median of the residuals is calculated, 
            along with the 25th percentile and the 75th percentile. 
            The difference between the 25th and 75th percentile is the interquartile deviation (IQD). 
            Then, the difference is calculated between each historical value and the residual median. 
            If the historical value is a certain number of MAD away from the median of the residuals, 
            that value is classified as an outlier. 
            The default threshold is 2.22, which is equivalent to 3 standard deviations or MADs.
            This method is somewhat susceptible to influence from extreme outliers, 
            but less so than the mean and standard deviation method.
            This method can be used for both symmetric and asymmetric data.
            
        Percentile Outlier Filtering (percentile) - A percentile indicates the value below which a given percentage 
            of observations in a group of observations fall.
            By default this method will remove all rows associated with a column with 
            values below 5% and above 95% percentiles
    
    Parameters:
    -----------
    dataset : Pandas DataFrame
        The dataset to remove outliers from
    cat_freq : sequence / string. default = 'all'
        A sequence of the nominal (categorical) columns in the dataset. If
        string, must be 'all' to state that all columns are nominal. If None,
        nothing happens. If 'auto', categorical columns will be identified
        based on dtype.
    std_threshold : Boolean, default = False
        If True, nominal columns with a only a single value will be dropped.
    lb_percentile : Boolean, default = True
        If True, the return value will be the encoded DataFrame alone. If
        False, it will be a tuple of the DataFrame and the dictionary of the
        binary factorization (originating from pd.factorize)
    ub_percentile : string, default = 'replace'
        How to handle missing values: can be either 'drop_samples' to remove
        samples with missing values, 'drop_features' to remove features
        (columns) with missing values, or 'replace' to replace all missing
        values with the nan_replace_value. Missing values are None and np.nan.
    lb_quantile : any, default = 0.0
        The value used to replace missing values with. Only applicable when nan
        _strategy is set to 'replace'
    ub_quantile : any, default = 0.0
        The value used to replace missing values with. Only applicable when nan
        _strategy is set to 'replace'
    num_method : str
        string describing the method to be used for numerical variables from available set ['SD', 'IQD', 'percentile']
    
    Returns:
    --------
    dataset : Pandas DataFrame
        The dataset with outliers removed
    report : Pandas DataFrame
        The report describing number of outliers removed
    """
    
    # raise exception for incorrect method selection
    if num_method not in ['SD', 'IQD', 'percentile']:
        raise Exception('[ERROR] please provide a valid method from the set (SD, IQD, percentile)')
    num_method
    # data reporting dictionary
    outliers_removed = dict()
    # identify categorical columns
    categorical_columns = identify_columns_by_type(dataset, include=['object', 'category', 'string'])
    timestamp_columns = identify_columns_by_type(dataset, include=['datetime64[ns]'])
    exclusions = timestamp_columns + exclude
    # iterate through each categorical column to remove outliers
    for col_name in dataset.columns:
        # size of current sample
        curr_sample = dataset.shape[0]
        # lower and upper bound
        minimum = -np.inf
        maximum = np.inf
        try:
            if col_name in categorical_columns and col_name not in exclusions:
                dataset[col_name] = dataset[col_name].str.title()
                dataset = dataset.groupby(col_name, dropna=False).filter(lambda x : len(x) >= cat_freq)
                minimum = cat_freq
            elif col_name not in exclusions and num_method == 'percentile':
                quant_series = dataset[col_name].quantile([lb_percentile, ub_percentile])
                dataset = dataset.loc[(dataset[col_name] >= quant_series[lb_percentile]) 
                                      & (dataset[col_name] <= quant_series[ub_percentile]) 
                                      | (dataset[col_name].isna())]
                minimum = quant_series[lb_percentile]
                maximum = quant_series[ub_percentile]
            elif col_name not in exclusions and num_method == 'SD':
                mean = dataset[col_name].mean()
                sd = dataset[col_name].std()
                dataset = dataset[(dataset[col_name] >= mean - (std_threshold * sd)) 
                                  & (dataset[col_name] <= mean + (std_threshold * sd)) 
                                  | (dataset[col_name].isna())]
                minimum = mean - (std_threshold * sd)
                maximum = mean + (std_threshold * sd)
            elif col_name not in exclusions and num_method == 'IQD':
                # Calculate quantiles and IQR
                Q1 = dataset[col_name].quantile(lb_quantile)
                Q3 = dataset[col_name].quantile(ub_quantile)
                IQR = Q3 - Q1
                # Filter our dataframe based on condition
                dataset = dataset.loc[(dataset[col_name] >= (Q1 - 1.5 * IQR)) 
                                      & (dataset[col_name] <= (Q3 + 1.5 * IQR)) 
                                      | (dataset[col_name].isna())]
                minimum = (Q1 - 1.5 * IQR)
                maximum = (Q3 + 1.5 * IQR)
        except Exception as e:
            logger.error(f'[ERROR] unknwon error occured when processing column {col_name}, debugging in progress...')
            logger.error('[ERROR] Oops! Failed to process', e)
            continue
        records_removed = curr_sample - dataset.shape[0]
        # populate outlier reporting dictionary
        outliers_removed.setdefault(col_name, []).append(records_removed)
        outliers_removed.setdefault(col_name, []).append(minimum)
        outliers_removed.setdefault(col_name, []).append(maximum)
        # convert to Data Frame
        report = pd.DataFrame.from_dict(outliers_removed, 
                                        orient='index', 
                                        columns=['Outliers', 'Lower Bound', 'Upper Bound'])
        # rename index column
        report.index.names = ['Feature']
        report.reset_index(inplace=True)
        # highlight 'Feature' column
        report = display(report,
                                   rows=report.shape[0],
                                   color='green',
                                   columns_to_shadow=['Feature'],
                                   columns_to_show=[])

    return dataset, report

class LabelEncoderCustom():

    """
    1. Encoding a dataset with mixed data (numerical and categorical) to a
    numerical-only dataset using the following logic:
    * categorical with only a single value will be marked as zero (or dropped,
        if requested)
    * categorical with two values will be replaced with the result of Pandas
        'factorize'
    * categorical with more than two values will be replaced with the result
        of Pandas 'get_dummies'
    * numerical columns will not be modified
    
    2. Transform a dataset with mixed data (numerical and categorical) to a
    numerical-only dataset based on learned encodings
    
    3. Inverse Transform encoded numerical-only dataset to the originally labeled 
    mixed data (numerical and categorical)
    
    Attributes:
    -----------
    binary_columns_mapping : dict()
        Stores binary column label to numerical value mapping for transformation
    ordinal_columns_mapping : dict()
        Stores ordinal column label to numerical value mapping for transformation
        
    Examples:
    ---------
    LabelEncoderCustom to transform non-numerical labels
    
    >>> from elasticdatafactory.utilities import utility
    >>> cle = utility.LabelEncoderCustom()
    >>> dataset = pd.DataFrame({
                        'A': ['a', 'b', 'a'], 
                        'B': ['b', 'a', 'c'],
                        'C': [1, 2, 3]
                        })
    >>> cle.fit_transform(dataset,
                          categorical_columns='auto',
                          ordinal_columns=['B'],
                          cat_order={'B': ['c','a','b']},
                          drop_single_label=True,
                          drop_fact_dict=True,
                          nan_strategy='drop',
                          nan_replace_value=0)
        
    pandas.core.frame.DataFrame
        A	B	C
    0	0	2	1
    1	1	1	2
    2	0	0	3
    
    >>> cle.transform(pd.DataFrame({'A': ['a', 'b', 'a'], 'C': [1, 2, 3]}))
    
    pandas.core.frame.DataFrame
        A	C
    0	0	1
    1	1	2
    2	0	3
    
    >>> cle.inverse_transform({'A': [0, 1, 0], 'C': [1, 2, 3]})
  
    pandas.core.frame.DataFrame
        A	C
    0	a	1
    1	b	2
    2	a	3
    
    >>> cle.get_params()
    
    {'A': {0: 'a', 1: 'b'}, 'B': ['B_a', 'B_b', 'B_c']}
        
    """

    def __init__(self):
        
        # transformation mappings
        self.dataset_columns = list()
        self.binary_mapping = dict() 
        self.ordinal_mapping = dict()
        self.nominal_mapping = dict()

    def fit_transform(self, dataset, categorical_columns='auto', ordinal_columns=[], cat_order=None, 
                      drop_single_label=False, nan_strategy='replace', nan_replace_value=0):
    
        """
        Encoding a dataset with mixed data (numerical and categorical) to a 
        numerical-only dataset
        
        Parameters
        ----------
        self : object
            LabelEncoderCustom class instance.
        dataset : NumPy ndarray / Pandas DataFrame
            The dataset to fit encodings and transform
        categorical_columns : sequence / string. default = 'all'
            A sequence of the dichotomous + nominal + ordinal (categorical) columns in the dataset. If
            string, must be 'all' to state that all columns are nominal. If None,
            nothing happens. If 'auto', categorical columns will be identified
            based on dtype.
        ordinal_columns : list
            A sequence of the ordinal (categorical) columns in the dataset. If None,
            all categorical columns are considered nominal based on defined rules and dtype.
        cat_order : dict()
            A dictionary of ordinal (categorical) values in the dataset specifying 
            clear ordering of the categories. If None, default ordering will be used. For instance,
            {'ordinal_column_A': ['ordinal value A', 'ordinal value B', 'ordinal value C']}
        drop_single_label : Boolean, default = False
            If True, nominal columns with a only a single value will be dropped.
        nan_strategy : string, default = 'replace'
            How to handle missing values: can be either 'drop_samples' to remove
            samples with missing values, 'drop_features' to remove features
            (columns) with missing values, or 'replace' to replace all missing
            values with the nan_replace_value. Missing values are None and np.nan.
        nan_replace_value : any, default = 0.0
            The value used to replace missing values with. Only applicable when nan
            _strategy is set to 'replace'
    
        Returns:
        --------
        transformed_dataset : Pandas DataFrame
            Returns the transformed numerical-only Pandas DataFrame
        """
    
        # reset transformation mappings
        self.dataset_columns = list()
        self.binary_mapping = dict() 
        self.ordinal_mapping = dict()
        self.nominal_mapping = dict()
        # create an empty Pandas DataFrame to populate with transformed data
        transformed_dataset = pd.DataFrame()
        # convert data to Pandas DataFrame
        dataset = convert_data_structure(dataset, 'DataFrame')
        # check if dataset is None
        if dataset is None:
            logger.error(f'[ERROR] provided dataset is not a compatible data structure!')
            raise ValueError('[ERROR] error encountered, exiting label encoder...')
        # populate columns
        self.dataset_columns = dataset.columns
        if nan_strategy == 'replace':
            if 'string' in list(dataset.dtypes) and type(nan_replace_value) != str:
                nan_replace_value = str(nan_replace_value)
                logger.info(f'[INFO] nan_replace_value <{nan_replace_value}> is transformed to a string data type')
            dataset.fillna(nan_replace_value, inplace=True)
        elif nan_strategy == 'drop':
            dataset.dropna(axis=0, inplace=True)
        elif nan_strategy == 'drop_features':
            dataset.dropna(axis=1, inplace=True)
        # check validity of ordinal_columns assigned value
        if ordinal_columns not in [None] and isinstance(ordinal_columns, list) != True:
            logger.info(f'[INFO] valid input for <ordinal_columns> is a <list of ordinal columns> or <None>')
            logger.error(f'[ERROR] provided ordinal_columns <{ordinal_columns}> is invalid, '\
                         'ignoring assigned value...')
            ordinal_columns = list()
        elif ordinal_columns is None:
            ordinal_columns = list()
        # check validity of cat_order assigned value
        if isinstance(cat_order, dict) != True and cat_order is not None:
            logger.error("[ERROR] following default order since cat_order has an unknown data structure, "\
                         "please use <None> or the following schema " \
                         "{'ordinal_column_A': ['ordinal value A', 'ordinal value B', 'ordinal value C']}")
            cat_order = dict()
        elif cat_order is None:
            cat_order = dict()
        # check validity of categorical_columns assigned value
        if categorical_columns not in ['auto', 'all', None] and isinstance(categorical_columns, list) != True:
            logger.info(f'[INFO] valid values for <categorical_columns> '
                        'are either <None>, <auto>, <all>, or <list of nominal columns>')
            logger.error(f'[ERROR] provided categorical_columns <{categorical_columns}> '\
                         'is invalid, executing encoder in auto mode...')
            categorical_columns = 'auto'
        elif isinstance(categorical_columns, list) == True or categorical_columns == None:
            logger.info(f'[INFO] using user-defined categorical columns for label encoding...')
            if categorical_columns == None:
                categorical_columns = list()
        # encode all columns
        if categorical_columns == 'all':
            categorical_columns = dataset.columns
        # encode only categorical columns in dataset
        elif categorical_columns == 'auto':
            categorical_columns = identify_columns_by_type(dataset, include=['object', 'category'])
        # iterate through dataset columns
        for col in tqdm(dataset.columns):
            if col not in categorical_columns:
                # column not encoded
                transformed_dataset.loc[:, col] = dataset[col]
            elif col not in categorical_columns and col in ordinal_columns:
                # generate warning
                logger.warning(f'[WARNING] ordinal column {col} is not specified as categorical, '\
                               'skipping label encoding...')
                # column not encoded
                transformed_dataset.loc[:, col] = dataset[col]
            else:
                # count unique entries
                unique_values = pd.unique(dataset[col])
                # column with zero variance assigned 0 if not to be dropped
                if len(unique_values) == 1 and not drop_single_label:
                    transformed_dataset.loc[:, col] = 0
                # apply binary encoding for binary variables
                elif len(unique_values) == 2 and col not in ordinal_columns:
                    binary_data = dataset[col].astype(pd.api.types.CategoricalDtype(categories=None))
                    transformed_dataset.loc[:, col] = binary_data.cat.codes
                    self.binary_mapping[col] = dict(zip(transformed_dataset.loc[:, col], dataset[col]))
                # apply integer encoding to ordinal columns
                elif len(unique_values) >= 2 and col in ordinal_columns:
                    # extract column order
                    cat_order_col = cat_order.get(col)
                    # check if an empty order list
                    if isinstance(cat_order_col, list) == True and len(cat_order_col) == 0:
                        cat_order_col = None
                    try:
                        if cat_order_col is None:
                            logger.warning(f'[WARNING] default ordering is used for ordinal column {col}')
                        ordinal_data = dataset[col].astype(pd.api.types.CategoricalDtype(categories=cat_order_col))
                        transformed_dataset.loc[:, col] = ordinal_data.cat.codes
                        self.ordinal_mapping[col] = dict(zip(transformed_dataset.loc[:, col], dataset[col]))
                    except Exception as e:
                        logger.error(f'[ERROR] unknwon error occured when encoding column {col}, debugging in progress...')
                        logger.error(f'[ERROR] error caught: {e}')
                    
                # apply one-hot encoding to nominal columns
                else:
                    dummies = pd.get_dummies(dataset[col], prefix=col)
                    # estimator for one-hot encoding
                    self.nominal_mapping[col] = dummies.columns.tolist()
                    transformed_dataset = pd.concat(
                        [transformed_dataset, dummies], axis=1
                    )
        # return only encoded dataset
        return transformed_dataset
        
    def get_params(self):
        
        """
        This function is used to get parameters for the estimator and
            contained subobjects that are estimators.
        
        Parameters
        ----------
        self : object
            LabelEncoderCustom class instance.
        
        Returns
        -------
        estimated_parameters : dict
            A dictionary, where each key is column, and the value is a dictionary with key as 
            transformed numerical-only value and value as an original label in a given  categorical 
            column with the exception of one-hot encoded columns where value is a list of 
            one-hot encoded features. Keys will only be present for the transformed columns
        """
        
        # instantiate param dictionary
        estimated_parameters = dict()
        # assign value to param
        estimated_parameters = {**self.binary_mapping, **self.ordinal_mapping, **self.nominal_mapping}
        
        return estimated_parameters
        
    def transform(self, dataset):
        
        """
        Transform a dataset with mixed data (numerical and categorical) to a
        numerical-only dataset based on learned encodings.
        
        Parameters
        ----------
        self : object
            LabelEncoderCustom class instance.
        dataset : NumPy ndarray / Pandas DataFrame
            The dataset to transform based on learned encodings
    
        Returns:
        --------
        transformed_dataset : Pandas DataFrame
            The transformed numerical-only Pandas DataFrame from mixed data (numerical and categorical)
        """
        
        # check if dataset is None
        if dataset is None:
            logger.error(f'[ERROR] provided dataset is not a compatible data structure!')
            raise ValueError('[ERROR] error encountered, exiting label encoder...')
        # convert data to Pandas DataFrame
        dataset = convert_data_structure(dataset, 'DataFrame')
        # create an empty Pandas DataFrame to populate with transformed data
        transformed_dataset = pd.DataFrame()
        # aggregate feature-value pairs
        columns_mappings = {**self.binary_mapping, **self.ordinal_mapping}
        # iterate through all columns
        for col in tqdm(list(dataset.columns)):
            if col in self.binary_mapping or col in self.ordinal_mapping:
                replace_from = list(columns_mappings[col].values())
                replace_to = list(columns_mappings[col].keys())
                dataset[col] = dataset[col].astype('str')
                transformed_dataset[col] = dataset[col].replace(replace_from, replace_to)
            # apply one-hot encoding to nominal columns
            elif col in self.nominal_mapping:
                dummies = pd.get_dummies(dataset[col], prefix=col)
                # ensure integrity of data
                dummies = dummies.reindex(columns=self.nominal_mapping[col], fill_value=0)
                # cocatenate one-hot encoded features
                transformed_dataset = pd.concat(
                    [transformed_dataset, dummies], axis=1
                )
            # perform no transformation
            else:
                transformed_dataset[col] = dataset[col]
        # return transformed data using learned embeddings
        return transformed_dataset
    
    def transform_from_estimators(self, dataset, estimated_parameters):
        
        """
        The method works with estimators to transform dataset that was previously fitted.
        
        Parameters
        ----------
        self : object
            LabelEncoderCustom class instance.
        dataset : NumPy ndarray / Pandas DataFrame
            The dataset to fit encodings and transform
        estimated_parameters : dict
            A dictionary, where each key is column, and the value is a dictionary with key as 
            transformed numerical-only value and value as an original label in a given  categorical 
            column with the exception of one-hot encoded columns where value is a list of 
            one-hot encoded features. Keys will only be present for the transformed columns
        
        Returns:
        --------
        transformed_dataset : Pandas DataFrame
            The transformed numerical-only Pandas DataFrame from mixed data (numerical and categorical)
            
        """
        
        # transformation mappings
        self.dataset_columns = list()
        self.binary_mapping = dict() 
        self.ordinal_mapping = dict()
        self.nominal_mapping = dict()
        
        # check if dataset is None
        if dataset is not None:
            self.dataset_columns = list(dataset.columns)
        if estimated_parameters is None:
            logger.error(f'[ERROR] provided estimators do not follow a compatible schema!')
            raise ValueError('[ERROR] error encountered, exiting label encoder...')
        for key in list(estimated_parameters.keys()):
            if isinstance(estimated_parameters[key], list) == True:
                self.nominal_mapping[key] = estimated_parameters[key]
            else:
                self.binary_mapping[key] = estimated_parameters[key]
        # transform data
        transformed_dataset = self.transform(dataset)
        # return transformed data using learned embeddings
        return transformed_dataset
    
    def get_col_name(self, row, col, subset_col):
        
        """
        Inverse Transform encoded numerical-only dataset to the originally labeled 
        mixed data (numerical and categorical)
        
        Parameters
        ----------
        self : object
            LabelEncoderCustom class instance.
        """
        
        # iterate through column values per row
        for hot_one_col in subset_col:
            if row[hot_one_col] == 1:
                return hot_one_col.replace(col,'')[1:]
        
    def inverse_transform(self, dataset):
        
        """
        Inverse Transform encoded numerical-only dataset to the originally labeled 
        mixed data (numerical and categorical)
        
        Parameters
        ----------
        self : object
            LabelEncoderCustom class instance.
        dataset : NumPy ndarray / Pandas DataFrame
            The dataset to inverse transform for generation of orginal labels
    
        Returns:
        --------
        inverse_transformed_dataset : Pandas DataFrame
            The mixed data (numerical and categorical) from transformed numerical-only Pandas DataFrame
        """
        
        # check if dataset is None
        if dataset is None:
            logger.error(f'[ERROR] provided dataset is not a compatible data structure!')
            raise ValueError('[ERROR] error encountered, exiting label encoder...')
        # convert data to Pandas DataFrame
        dataset = convert_data_structure(dataset, 'DataFrame')
        # aggregate binary and ordinal dictionaries
        columns_mappings = {**self.binary_mapping, **self.ordinal_mapping}
        # replace values based on aggregated dictionary
        inversed_dataset = dataset.replace(columns_mappings)
        # reverse one-hot encoding if applicable
        if len(self.nominal_mapping) > 0:
            for col in tqdm(list(self.nominal_mapping.keys())):
                subset_col = list(filter(lambda x: x.startswith(col), self.nominal_mapping[col]))
                try:
                    inversed_dataset[col] = inversed_dataset[self.nominal_mapping[col]].apply(
                        lambda x: self.get_col_name(x, col, subset_col), axis=1)
                except Exception as e:
                    logger.error('[ERROR] unknwon error occured, debugging in progress...')
                    logger.error(f'[ERROR] error caught, unable to inverse an unknown transformation: {e}') 
            # drop one-hot inverse columns
            inversed_dataset = inversed_dataset.drop(self.nominal_mapping[col], axis=1)
            
        return inversed_dataset[self.dataset_columns]    
    
def data_partition(data, target=None, sequential_data=False, timestamp=None, 
                   n_splits_val=5, n_splits_test=3, gap=0, group=None, stratify=False, shuffle=False):
    
    """
    This function splits the dataset into features and target variable among train/validation/test

    Parameters:
    -----------
    data : Pandas Dataframe
        input data
    target : str
        name of the target variable
    sequential_data : boolean, default False
        check if the data is sequential
    timestamp : str, default None
        column name of the timestamp in dataset
    n_splits_val : int
        number of partitions for sequential data
    n_splits_test : int
        number of KFold for non-sequential data
    gap : int, default 0
        gap between partitions to simulate delays between development and deployment timeline
    group : str, default None
        group samples by this value to be split
    stratify : boolean, default False
        used in the StratifiedKFold and StratifiedGroupKFold
    shuffle : boolean
        whether to shuffle the data before splitting into batches

    Returns:
    --------
    X_train : Pandas Dataframe
        independent variables for the train dataset
    X_val : Pandas Dataframe
        independent variables for the validation dataset
    X_test : Pandas Dataframe
        independent variables for the test dataset
    y_train : Pandas Series
        target variable for the train dataset
    y_val : Pandas Series
        target variable for the validation dataset
    y_test : Pandas Series
        target variable for the test dataset
    """
    # set up cross validators from scikit-learn
    cross_vals_type = [KFold,
                       GroupKFold,
                       ShuffleSplit,
                       StratifiedKFold,
                       StratifiedGroupKFold,
                       GroupShuffleSplit,
                       StratifiedShuffleSplit,
                       TimeSeriesSplit
                      ]
    # set random state
    if shuffle:
        random_state=76920
    else:
        random_state=None
    # create a copy of the dataFrame
    data = data.copy()
    # check if data_type is valid
    if sequential_data != True and sequential_data != False:
        logger.error(f'[ERROR] data type <{sequential_data}> is invalid, ignoring sequential dependency...')
        sequential_data = False
    # check if timestamp is valid
    if sequential_data and timestamp in data.columns.to_list():
        data.set_index(timestamp, inplace=True)
        data.sort_index(inplace=True)
    elif sequential_data and timestamp not in data.columns.to_list():
        logger.error(f'[ERROR] provided timestamp <{timestamp}> is invalid, try again with a valid timestamp!')
        return
    # check if target is valid
    if target not in data.columns.to_list():
        logger.error(f'[ERROR] provided target <{target}> is invalid, try again with a valid target!')
        return
    # split data into train and test
    X = data.drop(labels=[target], axis=1)
    y = data[target]
    # check if group is valid
    if group not in data.columns.to_list() and group != None:
        logger.error(f'[ERROR] provided group <{group}> is invalid, skipping group partitioning...')
    # partition data by group
    if group in data.columns.to_list():
        # convert group to integers using integer encoding as a grouping parameter for partitioning
        group_integers, labels = pd.factorize(data[group])
    # select a cross validator
    if sequential_data:
        # select cross validator as TimeSeriesSplit
        cv = TimeSeriesSplit(gap=gap, max_train_size=None, n_splits=n_splits_test, test_size=None)
    else:
        cv = KFold(n_splits=n_splits_test, shuffle=shuffle, random_state=random_state)
    # plot data split distribution by class and group
    plotter.plot_cv_indices(cv=cv, X=X, y=y, group=group_integers, n_splits=n_splits_test)

    # *** train on the last partition only ***
    for train_val_index, test_index in cv.split(X=X, y=y, groups=group_integers):
        logger.info(f'TRAIN & VAL:, {len(train_val_index)} ({round(len(train_val_index)/X.shape[0]*100)}%), ' \
                    f'TEST:, {len(test_index)} ({round(len(test_index)/X.shape[0]*100)}%)')
        X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
        y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]
        
    # partition data by group
    if group in X_train_val.columns.to_list():
        # convert group to integers using integer encoding as a grouping parameter for partitioning
        group_integers, labels = pd.factorize(X_train_val[group])
    if sequential_data:
        cv = TimeSeriesSplit(gap=gap, max_train_size=None, n_splits=n_splits_val, test_size=None)
    else:
        cv = KFold(n_splits=n_splits_test, shuffle=shuffle, random_state=random_state)
    # plot data split distribution by class and group
    plotter.plot_cv_indices(cv, X=X_train_val, y=y_train_val, group=group_integers, n_splits=n_splits_val)

    # *** train and validate on the last partition only ***
    for train_index, val_index in cv.split(X=X_train_val, y=y_train_val, groups=group_integers):
        logger.info(f'TRAIN:, {len(train_index)} ({round(len(train_index)/X.shape[0]*100)}%), ' \
                    f'VALIDATION:, {len(val_index)} ({round(len(val_index)/X.shape[0]*100)}%)')
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
def display(df, rows=20, color='lightgreen', columns_to_shadow=[], columns_to_show=[]):
    
    """Function to display and highlight Pandas DataFrame columns based on headers
    
    Parameters
    ----------
    df : Pandas DataFrame
        data stored in dataFrame
    rows : int    
        specify number of rows to display
    color : str
        specify highlight color
    columns_to_shadow : list
        A list containing column names to highlight
    columns_to_show : list
        A list containing column names to display. It will display
        all columns if left empty

    Returns
    -------
    highlighted_df : Pandas DataFrame
        Return highlighted Pandas DataFrame with a subset of columns
    """
    
    highlight = lambda slice_of_df: 'background-color: %s' % color
    sample_df = df.head(rows)
    # check if columns_to_show is empty
    if len(columns_to_show) != 0:
        # check if user provided columns exist
        columns_to_show = list(set(sample_df.columns.to_list()) & set(columns_to_show))
        if len(columns_to_show) != 0:
            sample_df = sample_df[columns_to_show]
    # check if user provided columns exist
    columns_to_shadow = list(set(sample_df.columns.to_list()) & set(columns_to_shadow))
    highlighted_df = sample_df.style.applymap(highlight, subset=pd.IndexSlice[:, columns_to_shadow])
    
    return highlighted_df

def compute_cramers_v(x, y, bias_correction=True, nan_strategy='replace', nan_replace_value=0.0, precision=1e-13):
    
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    This is a symmetric coefficient: V(x,y) = V(y,x)

    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
        Use bias correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328.
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.
    
    Returns:
    --------
    cramers_v : float in the range of [0,1]
        Return Cramer's V value for the two nominal variables provided
    """
    if nan_strategy == 'replace':
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == 'drop':
        x, y = remove_incomplete_samples(x, y)
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if bias_correction:
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        if min((kcorr - 1), (rcorr - 1)) == 0:
            logger.warning("Unable to calculate Cramer's V using bias correction. " \
                          "Consider using bias_correction=False (or cramers_v_bias_correction=False " \
                          "if calling from associations)"
                         )
            return np.nan
        else:
            cramers_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    else:
        if min(k - 1, r - 1) == 0:
            return np.nan
        cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))
    if -precision <= cramers_v < 0.0 or 1.0 < cramers_v <= 1.0 + precision:
        rounded_cramers_v = 0.0 if cramers_v < 0 else 1.0
        logger.warning(
            f"Rounded Cramer's V = {cramers_v} to {rounded_cramers_v}. This is probably due to floating point precision issues."
        )
        return rounded_cramers_v
    else:
        return cramers_v

def hyperparam_tuning_stats(tuning_job_name, sagemaker_client, visualize=False):
    
    """
    Function to display and highlight sagemaker hyperparmeter tuning statistics
    
    Parameters
    ----------
    tuning_job_name : str    
        Specify the name of hyperparameter tuning jon on Sagemaker
    sagemaker_client : boto3.client('sagemaker')
        A low-level client representing Amazon SageMaker Service
    visualize : bool
        This will visualize hyperparameter tuning results if set to True. Default to False

    Returns
    -------
    df : Pandas DataFrame
        Return highlighted Pandas DataFrame with all results from the defined tuning job
    """
    
    # check current status of hyperparameter tuning job
    tuning_job_result = sagemaker_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
    )
    status = tuning_job_result['HyperParameterTuningJobStatus']
    if status != 'Completed':
        logger.info('[INFO] Reminder: the tuning job has not completed yet!')
        return

    job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
    logger.info(f'[INFO] {job_count} training jobs have completed')

    if tuning_job_result.get('BestTrainingJob', None):
        print('[INFO] Best model found so far:')
        pprint(tuning_job_result['BestTrainingJob'])
    else:
        logger.info('[INFO] No training jobs have reported results yet!')
        return
    # report all results from hyperparameter tuning job
    tuning_analytics = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
    # create pandas dataframe to store results
    results_df = tuning_analytics.dataframe()
    # obtain objective metric attributes
    objective_metric = tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']
    objective_minimize = objective_metric['Type'] != 'Maximize'
    objective_metric_name = objective_metric['MetricName']
    
    if len(results_df) > 0:
        df = results_df[results_df['FinalObjectiveValue'] > -float('inf')]
        if len(df) > 0:
            df = df.sort_values('FinalObjectiveValue', ascending=objective_minimize)
            print(f'Number of training jobs with valid objective: {len(df)}')
            print({'lowest': min(df['FinalObjectiveValue']), 'highest': max(df['FinalObjectiveValue'])})
            print('\n')
            pd.set_option('display.max_colwidth', None)  # prevent truncation of values
            if visualize == True:
                plotter.PlotTuningProgress(df=df, tuning_job_name=tuning_job_name, sagemaker_client=sagemaker_client).plot_params()
            return df
        else:
            logger.info('No training jobs have reported valid results yet!')
            return
        
def load_model(bucket, key, s3_client):
    
    """
    This function deserializes and returns fitted model.
    Note that this should have the same name as the serialized model in the _xgb_train method
    
    Parameters
    ----------
    bucket : str
        S3 Bucket name containing objects
    key : str
        Key of the object to retrieve from S3 bucket
    s3_client : botocore.client.S3
        boto3 client name. E.g., boto3.client('s3')

    Returns
    -------
    model : bytes-like object
        Returns the reconstituted object hierarchy of the pickled representation data of an object.
    """
    
    # Retrieves objects from Amazon S3.
    try:
        s3_model = s3_client.get_object(Bucket=bucket, Key=key)['Body'].read()
    except Exception as e:
        logger.error('[ERROR] unknwon error occured, debugging in progress...')
        logger.error(f'[ERROR] error caught: {e}')
        return
    # uncompress tar file
    with tarfile.open(fileobj = BytesIO(s3_model)) as tar:
        for tar_resource in tar:
            if (tar_resource.isfile()):
                s3_model_file_bytes = tar.extractfile(tar_resource).read()
                # 'loads' is used to load pickled data from a bytes string as opposed to 'load' in file-like object
                model = pkl.loads(s3_model_file_bytes)
                logger.info('[INFO] model loaded successfully!')
                return model
        
class SagemakerModelWrapper(mlflow.pyfunc.PythonModel):
    
    """
    The following class creates a wrapper function, SagemakerModelWrapper, that uses
    the predict() method to return the probability that the observation belongs to a specific class
    
    Attributes
    ----------
    mlflow.pyfunc.PythonModel : mlflow.pyfunc module
        An MLflow wrapper around the model implementation 
        
    """
    
    def __init__(self, model):
        self.model = model

    def predict(self, context: mlflow.pyfunc.PythonModelContext, data):
        
        """
        Function to determine the probability that the observation belongs to a specific class
    
        Parameters
        ----------
        self : object    
            SagemakerModelWrapper class instance.
        context : mlflow.pyfunc method
            A collection of artifacts that a PythonModel uses when performing inference
        data : Pandas DataFrame
            Pandas DataFrame containing data used for model training

        Returns
        -------
        data : NumPy ndarray
            The predict method returns probabilities to be used for label classification.
        """
        
        # transform data to DMatrix format
        payload = xgb.DMatrix(data)
        # You don't have to keep the semantic meaning of `predict`
        return self.model.predict(payload)
    
    def predict_batch(self, data):
        
        """
        Extra functions if needed. Since the model is serialized,
        all of them will be available when the model is loaded.
    
        Parameters
        ----------
        data : Pandas DataFrame
            Pandas DataFrame containing data used for model training

        Returns
        -------
        None
        
        """
        
        pass
    
def create_mlflow_trial(project_title: str, description=None, run_name=None, stage='None', model=None, model_data=None, 
                        model_version='', metrics={}, params={}, tags={}, artifacts=(), figures={}, dependencies='requirements.txt'):
    
    """
    Function to register experimenets/projects, individual runs/trials, and respective model artifacts to MLflow
    
    Parameters
    ----------
    project_title : str    
        Specify the title of an experiment or a project
    description : str
        Description of the trial/run under the experiment, Default to None
    run_name : str
        The name to give the MLflow Run associated with the project execution. Default to None
    stage : str
        Represents model's lifecycle stage, valid values are [Staging|Archived|Production|None]. Default to None
    model : bytes-like object
        Returns the reconstituted object hierarchy of the pickled representation data of an object. Default to None
    model_data : Pandas DataFrame
        Pandas DataFrame containing data used for model training. Default to None
    model_version : str
        Represents model version that is ready to be deployed on MLflow, Default to an empty string
    metrics : dict()
        Provide metrics and corresponding values from model evaluation. Default to an empty dictionary
    params : dict()    
        Provide metrics and corresponding values from model evaluation. Default to an empty dictionary
    tags : dict()
        Log a batch of tags for the current run, such as {'estimator_name': estimator_name, 'version': '2.2'}
    artifacts : tuple(local_dir: str, artifact_path: Optional[str] = None)
        This logs all the contents of a local directory as artifacts of the run. Default to an empty tuple
    figures : dict(figure, artifact_file: str)
        Provide figure name as key and corresponding figure as value to log a figure as an artifact. 
        The following figure objects are supported: 
            matplotlib.figure.Figure 
            plotly.graph_objects.Figure
    dependencies : str
        Provides a list of dependencies in the form of a path to a text file, such as requirements.txt
        Defaults to requirements.txt

    Returns
    -------
    None
    
    """
    
    try:
        mlflow.set_experiment(project_title)
        with mlflow.start_run(description=description,
                              run_name=run_name,
                              tags={'mlflow.source.git.commit': model_version,
                                    'mlflow.user': 'Jenna',
                                    'mlflow.source.name': 'Sagemaker'}) as run:
            if len(metrics) != 0:
                mlflow.log_metrics(metrics)
            if len(params) != 0:
                mlflow.log_params(params)
            if len(tags) != 0:
                mlflow.set_tags(tags)
            if len(figures) != 0:
                for key in figures:
                    mlflow.log_figure(figures[key], key)
            if len(artifacts) == 1:
                mlflow.log_artifacts(local_dir=artifacts[0])
            elif len(artifacts) == 2:
                mlflow.log_artifacts(local_dir=artifacts[0], artifact_path=artifacts[1])
            else:
                logger.info('[INFO] no artifacts are provided to log from a local directory')
            if model is not None and model_data is not None:
                # assuming first column is considered as target var
                X_train = model_data.iloc[:, 1:]
                wrapped_model = SagemakerModelWrapper(model)
                # Log the model with a signature that defines the schema of the model's inputs and outputs.
                # When the model is deployed, this signature will be used to validate inputs.
                signature = infer_signature(X_train, wrapped_model.predict(None, X_train))
                # MLflow contains utilities to create a conda environment used to serve models.
                # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
                try:
                    with open(dependencies) as text_file:
                        dependency = text_file.readlines()
                        dependency = [line[:-1] for line in dependency]
                except Exception as e:
                    dependency = []
                    logger.error('[ERROR] unknwon error occured when examining project dependencies, debugging in progress...')
                    logger.error(f'[ERROR] error caught: {e}')
                conda_env =  _mlflow_conda_env(additional_conda_deps=None,
                                               additional_pip_deps=dependency,
                                               additional_conda_channels=None,
                                              )
                mlflow.pyfunc.log_model('XGBoost_Model', python_model=wrapped_model, conda_env=conda_env, signature=signature)
    except Exception as e:
        logger.error('[ERROR] unknwon error occured when registering model artifacts, debugging in progress...')
        logger.error(f'[ERROR] error caught: {e}')
        return

    # register trained model to MLflow
    try:
        model_uri = 'runs:/{}/XGBoost_Model'.format(run.info.run_id)
        mv = mlflow.register_model(model_uri, project_title)
        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))
        print("Stage: {}".format(mv.current_stage))
    except Exception as e:
        logger.error('[ERROR] unknwon error occured when registering trained model, debugging in progress...')
        logger.error(f'[ERROR] error caught: {e}')
    
    try:
        # Transition an MLflow Model’s Stage
        if str(stage) not in ['Staging', 'Archived', 'Production', 'None']:
            logger.error(f'[ERROR] provided stage <{str(stage)}> is invalid, '\
                         'please select from: Staging, Archived, Production, None')
            return

            mlflow_client.transition_model_version_stage(name=mv.name,
                                                         version=mv.version,
                                                         stage=stage
                                                        )
    except Exception as e:
        logger.error('[ERROR] unknwon error occured when transitioning model lifecycle stage, debugging in progress...')
        logger.error(f'[ERROR] error caught: {e}')
        return

def get_group_idx(df, col, rsuffix):
    
    """Group row records by a specified column and collect row indices into a list within same group.
    
    Parameters
    ----------
    df : Pandas DataFrame
        Input dataframe
    col : str
        DataFrame column name
    rsuffix : str
        Suffix to use from right frame’s overlapping columns
    Returns
    -------
    df : Pandas DataFrame
        Output dataframe with '_groupidx' column which contains list of row indices under same group by
    """
    
    # this will generate null values due to groupby method
    df = df.join(df.groupby(col)[col].progress_apply(lambda x: list(x.index)), on=col, rsuffix=rsuffix)
    return df

def explode_id_cols(df, col_list):
    
    """Transform each element of a list-like in input column list to a row
    
    Parameters
    ----------
    df : Pandas DataFrame
        Input dataframe
    col_list : list
        List of dataframe column names
    
    Returns
    -------
    df : Pandas DataFrame
        Output dataframe with exploded rows
    """
    
    for col in col_list:
        df[col] = df[col].progress_apply(lambda x: ast.literal_eval(x.replace('\n',',')) if (type(x)==str and x[0]=='[') else x)
        df = df.explode([col])
    return df


def combine_id(x, id_map):
    
    """Collect unique ids from id columns into a list
    
    Parameters
    ----------
    x : Pandas DataFrame
        Input dataframe
    id_map : dict
        Dictionary of id column mappings
    
    Returns
    -------
    pd.Series
        Return a pandas series that contains unique id lists
    """
    
    output = {}
    for k, v in id_map.items():
        output[k] = [x for x in pd.unique(x[v].values.ravel('K')) if pd.isna(x) == False] 
    return pd.Series(output, index=list(id_map.keys()))


def get_unique_element(x, include_NA=False):
    
    """Collect unique elements from a series
    
    Parameters
    ----------
    x : pd.Series
        Input series
    include_NA : boolean
        Bool value to indicate if includes None in output. Default to False
    
    Returns
    -------
    list
        Return a list of unique elements
    """
    
    if include_NA:
        return [x for x in set(x)]
    return [x for x in set(x) if x != None]


def get_first_element(x):
    
    """Collect first not null element from a series
    
    Parameters
    ----------
    x : pd.Series
        Input series
    
    Returns
    -------
    anytype
        Return a value that is not null or None with any type
    """
    
    output = None
    for t in x:
        if t != None and pd.isnull(t) == False:
            return t
    return output


### Functions to transform datasets
class Transformation():
    
    """A class that stores functions to transform datasets
    
    Attributes
    ----------
    None 
    """   
    
    def __init__(self):
        pass
    
    def transform_account_df(self, df):

        """transform account create data and convert datetimes to EST and UTC 
        Parameters
        ----------
        df : Pandas DataFrame
            Input account create dataframe
        Returns
        -------
        df : Pandas DataFrame
            Return transformed dataframe
        """

        df['utcdatetime'] = pd.to_datetime(df['utcdatetime'].str[:19], format = '%Y-%m-%dT%H:%M:%S', utc = True)
        df['accountcreate_estdatetime'] = df['utcdatetime'].dt.tz_convert('America/New_York')
        df = df.rename(columns = {'recordid': 'accountcreate_recordid',
                                  'utcdatetime': 'accountcreate_utcdatetime'})
        col = ['accountcreate_utcdatetime',
               'accountcreate_estdatetime',
               'accountcreate_recordid',
               'rocketaccountid'] 
        return df[col]


    def transform_lead_df(self, df, loanpurpose=''):

        """transform lead submit data and convert datetimes to EST and UTC 
        
        Parameters
        ----------
        df : Pandas DataFrame
            Input lead submit dataframe
        loanpurpose : str
            Filter output with loan purpose 'Refinance' or 'Purchase'. Default to ''
        
        Returns
        -------
        df : Pandas DataFrame
            Return transformed dataframe
        """

        df['leadreceived_utcdatetime'] = pd.to_datetime(df['leadreceived_utcdatetime'].str[:19], format = '%Y-%m-%dT%H:%M:%S', utc = True)
        df['estdatetime'] = pd.to_datetime(df['estdatetime'].str[:19], format = '%m/%d/%Y %H:%M:%S').dt.tz_localize('America/New_York')
        df = df.rename(columns = {'recordid': 'leadsubmission_recordid',
                                  'leadreceived_utcdatetime': 'leadsubmissionreceived_utcdatetime',
                                  'estdatetime': 'leadsubmissionsent_estdatetime'})
        if loanpurpose:
            df = df[df['loanpurpose']==loanpurpose].reset_index(drop = True)
        col = ['leadsubmissionreceived_utcdatetime',
               'leadsubmissionsent_estdatetime',
               'leadsubmission_recordid',
               'loannumber',
               'adobevisitorid',
               'loanguid',
               'rmclientid',
               'loanpurpose', 
               'leadsystem',
               'leadsourcecategory',
               'leadtypecode',
               'isduplicate',
               'rlautotransfer',
               'email'] 
        return df[col]


    def transform_event_df(self, df, long_to_wide=True):

        """transform adobe event data and convert datetimes to EST and UTC
        
        Parameters
        ----------
        df : Pandas DataFrame
            Input adobe event dataframe
        long_to_wide : boolean
            If True, convert adobe event ids to columns. Default to True
        
        Returns
        -------
        df : Pandas DataFrame
            Return transformed dataframe
        """

        df['estdatetime'] = pd.to_datetime(df['date_time'].str[:19], format = '%Y-%m-%d %H:%M:%S').dt.tz_localize('America/New_York')
        df['utcdatetime'] = df['estdatetime'].dt.tz_convert('UTC')
        del df['date_time']

        df = df.groupby(['uniquevisitkey','mcvisid_visitorid','eventid'])[['estdatetime','utcdatetime']].min().reset_index()

        if long_to_wide:

            lead_form_event = ['204','206','250']
            event_flag_list = []
            output = df[['uniquevisitkey','mcvisid_visitorid']].drop_duplicates()

            for e, g in df.groupby('eventid'):
                g = g.rename(columns = {'eventid': 'is_event_' + e,
                                        'estdatetime': e + '_estdatetime',
                                        'utcdatetime': e + '_utcdatetime'})
                g['is_event_' + e] = g['is_event_' + e].progress_apply(lambda x: 1 if pd.isna(x) == False else 0)
                if e in lead_form_event:
                    event_flag_list.append('is_event_' + e)
                output = pd.merge(output, g, on= ['uniquevisitkey','mcvisid_visitorid'], how= 'left') 

            output['leadform_event'] = output[event_flag_list].max(axis = 1)
            return output
        else:
            return df


    def transform_leadform_input_df(self, df):

        """transform adobe leadform input data, convert datetimes to EST and map marketing channel ids with marketing channel names
        
        Parameters
        ----------
        df : Pandas DataFrame
            Input lead form users input dataframe
        
        Returns
        -------
        df : Pandas DataFrame
            Return transformed dataframe
        """

        marketing_channel_map = {'1': 'Paid Search',
                                 '2': 'Natural Search',
                                 '3': 'Display',
                                 '4': 'Email',
                                 '5': 'Affiliate Networks',
                                 '6': 'Direct Traffic',
                                 '7': 'EMPTY6',
                                 '8': 'Social',
                                 '9': 'Web Referral',
                                 '10': 'Direct Mail',
                                 '11': 'EMPTY5',
                                 '12': 'Radio',
                                 '13': 'Television',
                                 '14': 'Lead Buy',
                                 '15': 'Rate Tables',
                                 '16': 'BizDev Lead Generation',
                                 '17': 'Acquisitions',
                                 '18': 'Sponsorships and Promotions',
                                 '19': 'Uncategorized',
                                 '20': 'FOC Referral',
                                 '0': 'None'
                                }
        df['first_marketingchannel_finder'] = df['first_va_finder_id'].map(marketing_channel_map)
        df['first_marketingchannel_closer'] = df['first_va_closer_id'].map(marketing_channel_map)

        for col in ['min_visit_date_time','max_visit_date_time','min_landingpage_date_time','min_testexperiment_date_time','max_testexperiment_date_time',
                    'min_rocketaccountid_date_time','min_loannumber_date_time','min_loanguid_date_time']:
            df[col[:-9]+'estdatetime'] = pd.to_datetime(df[col].str[:19], format = '%Y-%m-%d %H:%M:%S').dt.tz_localize('America/New_York')
            del df[col]
        return df


    def transform_milestone_df(self, df, long_to_wide=False, long_to_wide_column='loanmilestone_groupname', utc_tz=True):

        """
        Transforms loan milestone data and convert timestamps to UTC
        
        Parameters
        ----------
        df : Pandas DataFrame
            Input loan milestone dataframe
        long_to_wide : boolean
            If True, covert long_to_wide_column to columns. Default to True
        long_to_wide_column : str
            Select 'loanmilestone_groupname' or 'loanmilestone_eventname' to transform to wide table
        utc_tz : boolean
            Specify if current timestamps need to be converted from ET and reported in UTC
        
        Returns
        -------
        df : Pandas DataFrame
            Return transformed dataframe
        """

        df['estdatetime'] = df['estdatetime'].dt.tz_localize('America/New_York')
        df['utcdatetime'] = df['estdatetime'].dt.tz_convert('UTC') 

        if long_to_wide:
            df_min_milestone = df.groupby(['loanidentifierdimsk', 'loanmilestone_groupname']).agg({'estdatetime': ['min', 'max'], 
                                                                                                   'utcdatetime': ['min','max']}).reset_index()
            df_min_milestone.columns = df_min_milestone.columns.to_flat_index().map(lambda x: '_'.join(x))
            df_min_milestone = df_min_milestone.rename(columns= {'loanidentifierdimsk_': 'loanidentifierdimsk',
                                                                 'loanmilestone_groupname_': 'loanmilestone_groupname'})
            groupby_columns = ['loanidentifierdimsk',
                               long_to_wide_column,
                               'estdatetime_min', 
                               'estdatetime_max', 
                               'utcdatetime_min', 
                               'utcdatetime_max']
            if utc_tz == False:
                df_min_milestone = df_min_milestone.drop(['utcdatetime_min', 
                                                          'utcdatetime_max'], axis=1)
                groupby_columns = ['loanidentifierdimsk', 
                                   long_to_wide_column, 
                                   'estdatetime_min', 
                                   'estdatetime_max']
            output = df[['loanidentifierdimsk']].drop_duplicates()
            for e, g in df_min_milestone[groupby_columns].groupby(long_to_wide_column):
                g = g.rename(columns = {long_to_wide_column: 'is_milestone_' + e,
                                        'estdatetime_min': e + '_estdatetime_min',
                                        'estdatetime_max': e + '_estdatetime_max',
                                        'utcdatetime_min': e + '_utcdatetime_min',
                                        'utcdatetime_max': e + '_utcdatetime_max'})
                g['is_milestone_' + e] = g['is_milestone_' + e].progress_apply(lambda x: 1 if pd.isna(x) == False else 0)
                output = pd.merge(output, g, on= ['loanidentifierdimsk'], how ='left') 
            return output
        else:
            df_min_milestone = df.groupby(['loanidentifierdimsk', 'loanmilestone_groupname']).agg({'estdatetime': ['min', 'max'], 
                                                                                                   'utcdatetime': ['min','max']}).reset_index()
            df_min_milestone.columns = df_min_milestone.columns.to_flat_index().map(lambda x: '_'.join(x))
            df_min_milestone = df_min_milestone.rename(columns= {'loanidentifierdimsk_': 'loanidentifierdimsk',
                                                                 'loanmilestone_groupname_': 'loanmilestone_groupname'})
            if utc_tz == False:
                df_min_milestone = df_min_milestone.drop(['utcdatetime_min', 'utcdatetime_max'], axis=1)
            output = df[['loanidentifierdimsk']].drop_duplicates()
            output = pd.merge(output, df_min_milestone, on= ['loanidentifierdimsk'], how= 'left')
            return output


    def transform_loanidentifier_df(self, df):
        """Map loanidentifierdimsk with first not null loannumber and loanguid
        
        Parameters
        ----------
        df : Pandas DataFrame
            Input loan identifier dataframe
        
        Returns
        -------
        df : Pandas DataFrame
            Return transformed dataframe
        """
        logger.info('[INFO] group rows and find first occurrence of loan number/loanguid...')
        df = df.groupby('loanidentifierdimsk').aggregate({'loannumber': lambda x: get_first_element(x),
                                                    'loanguid': lambda x: get_first_element(x)}).reset_index()
        logger.info('[INFO] convert loan number data type to string...')
        df['loannumber'] = df['loannumber'].progress_apply(lambda x: str(x) if pd.isna(x) == False else x)
        logger.info('[INFO] convert loanguid data type to string...')
        df['loanguid'] = df['loanguid'].progress_apply(lambda x: str(x) if pd.isna(x) == False else x)
        
        return df


### Functions to join datasets
class Join():
    
    """A class that stores functions to join datasets
    
    Attributes
    ----------
    None 
    
    """   
    
    def __init__(self):
        pass
    
    def join_preloan_id_with_purpose(self, df_id, df_purpose, loanpurpose=''):

        """Join preloan id with preloan purpose and transform datetimes to EST and UTC 
        
        Parameters
        ----------
        df_id : Pandas DataFrame
            Input preloan id dataframe
        df_purpose : Pandas DataFrame
            Input preloan purpose dataframe
        loanpurpose : str
            Filter output with loan purpose 'Refinance' or 'Purchase'. Default to ''
        
        Returns
        -------
        df : Pandas DataFrame
            Return joined dataframe
        """

        df = pd.merge(df_id, df_purpose, on = ['recordid', 'datekey']).reset_index(drop = True)
        df['utcdatetime'] = pd.to_datetime(df['utcdatetime'].str[:19], format = '%Y-%m-%dT%H:%M:%S', utc = True)
        df['preloan_estdatetime'] = df['utcdatetime'].dt.tz_convert('America/New_York')
        df = df.rename(columns = {'recordid': 'preloan_recordid',
                                  'utcdatetime': 'preloan_utcdatetime'})
        if loanpurpose:
            df = df[df['loanpurpose']==loanpurpose].reset_index(drop = True)
        col = ['preloan_utcdatetime',
               'preloan_estdatetime',
               'preloan_recordid',
               'loanguid',
               'loannumber',
               'rocketaccountid',
               'loanpurpose']     
        return df[col]


    def join_preloan_with_account(self, df_preloan, df_account, join='outer'):

        """Join preloan and account create on rocketaccountid and transform datetimes to EST and UTC 
        
        Parameters
        ----------
        df_preloan : Pandas DataFrame
            Input preloan dataframe
        df_account : Pandas DataFrame
            Input account create dataframe
        join : str
            Join method. Default to 'outer'
        
        Returns
        -------
        df : Pandas DataFrame
            Return joined dataframe
        """

        df = pd.merge(df_preloan, df_account, on = 'rocketaccountid', how = join)
        cols = ['rocketaccountid',
               'loanguid',
               'loannumber',
               'loanpurpose',
               'preloan_utcdatetime',
               'preloan_estdatetime',
               'preloan_recordid',
               'accountcreate_utcdatetime',
               'accountcreate_estdatetime',
               'accountcreate_recordid']
        
        return df[cols]


    def join_lead_with_preloan_with_account(self, df_lead, df_preloan, df_account):

        """Join lead, preloan and account create on loannumber and loanguid and transform datetimes to EST and UTC 
        
        Parameters
        ----------
        df_lead : Pandas DataFrame
            Input lead dataframe
        df_preloan : Pandas DataFrame
            Input preloan dataframe
        df_account : Pandas DataFrame
            Input account create dataframe
        
        Returns
        -------
        df : Pandas DataFrame
            Return joined dataframe
        """

        df_preloan_account = self.join_preloan_with_account(df_preloan, df_account)
        # lead form columns
        col1 = ['leadsubmissionreceived_utcdatetime', 'leadsubmissionsent_estdatetime', 'loannumber', 'adobevisitorid', 'loanguid', 'email']
        # preloan and account create table columns
        col2 = ['loannumber', 'loanguid', 'rocketaccountid', 'preloan_utcdatetime', 'preloan_estdatetime',
                'accountcreate_utcdatetime', 'accountcreate_estdatetime']

        df1 = pd.merge(df_lead[col1], 
                       df_preloan_account[(df_preloan_account['loannumber'].notna()) & (df_preloan_account['loanguid'].notna())][col2], 
                       on= ['loannumber','loanguid'], how= 'inner')
        df2 = pd.merge(df_lead[(df_lead['loannumber'].notna()) & (df_lead['loanguid'].notna())][col1], 
                       df_preloan_account[(df_preloan_account['loannumber'].isna()) & (df_preloan_account['loanguid'].notna())][col2[1:]], 
                       on= ['loanguid'], how= 'inner')
        df3 = pd.merge(df_lead[(df_lead['loannumber'].isna()) & (df_lead['loanguid'].notna())][col1], 
                       df_preloan_account[(df_preloan_account['loannumber'].isna()) & (df_preloan_account['loanguid'].notna())][col2[1:]], 
                       on= ['loanguid'], how= 'outer')
        df4 = df_lead[(df_lead['loannumber'].notna()) & (df_lead['loanguid'].isna())][col1]
        df5 = df_preloan_account[(df_preloan_account['loannumber'].isna()) & (df_preloan_account['loanguid'].isna())][col2[2:]]

        output = pd.concat([df1, df2, df3, df4, df5], axis=0)
        cols = ['loannumber',
                'adobevisitorid', 
                'loanguid', 
                'rocketaccountid',
                'email',
                'leadsubmissionreceived_utcdatetime', 
                'leadsubmissionsent_estdatetime',
                'preloan_utcdatetime',
                'preloan_estdatetime', 
                'accountcreate_utcdatetime',
                'accountcreate_estdatetime'
               ]
        
        return output.drop_duplicates(keep='first').reset_index(drop=True)


    def join_event_visitor_with_leadform_input(self, df_event, df_leadform_input, loanpurpose='', join='left'):

        """Join adobe event and leadform input data on uniquevisitkey and mcvisid_visitorid. 
           Replace '0' in rocketmortgage_loannumber_evar8 with pd.NA
        
        Parameters
        ----------
        df_leadform_input : Pandas DataFrame
            Input lead form user input dataframe
        df_event : Pandas DataFrame
            Input account create dataframe
        loanpurpose : str
            Filter output with first/last occured loan purpose 'Refinance' or 'Purchase'. Default to ''
        join : str
            Join method. Default to 'left'
        
        Returns
        -------
        df : Pandas DataFrame
            Return joined dataframe
        """

        if loanpurpose:
            df_leadform_input = df_leadform_input[(df_leadform_input['first_loanpurpose']==loanpurpose)|(df_leadform_input['last_loanpurpose']==loanpurpose)]

        df = pd.merge(df_leadform_input, df_event, on = ['uniquevisitkey','mcvisid_visitorid'], how = join)
        df['rocketmortgage_loannumber_evar8'] = df['rocketmortgage_loannumber_evar8'].replace('0', pd.NA)

        if 'leadform_event' in df.columns:
            df['leadform_event'] = df['leadform_event'].fillna(0)
        return df


    def join_lead_id_with_clickstream_lead_data(self, df_id, df_visit, unique_id):

        """Join adobe visit data with lead/loan ids by adobevisitorid, rocketaccountid, loannumber and loanguid
           Append unjoined adobe visit records to the end
        
        Parameters
        ----------
        df_id : Pandas DataFrame
            Input lead/loan id dataframe
        df_visit : Pandas DataFrame
            Input adobe visit dataframe
        unique_id : str
            Unique row identifier. If not input, set unique_id to index of dataframe
        
        Returns
        -------
        df : Pandas DataFrame
            Return joined dataframe
        """

        if not unique_id:
            df_visit = df_visit.reset_index()
            unique_id = 'index'

        df1 = pd.merge(df_id[df_id['adobevisitorid'].notna()], 
                       df_visit[df_visit['mcvisid_visitorid'].notna()], 
                       left_on = 'adobevisitorid', 
                       right_on = 'mcvisid_visitorid', 
                       how = 'inner')
        df2 = pd.merge(df_id[df_id['rocketaccountid'].notna()], 
                       df_visit[df_visit['rocketaccountid_evar5'].notna()], 
                       left_on = 'rocketaccountid', 
                       right_on = 'rocketaccountid_evar5', 
                       how = 'inner')
        df3 = pd.merge(df_id[df_id['loannumber'].notna()], 
                       df_visit[df_visit['rocketmortgage_loannumber_evar8'].notna()], 
                       left_on = 'loannumber', 
                       right_on = 'rocketmortgage_loannumber_evar8', 
                       how = 'inner')
        df4 = pd.merge(df_id[df_id['loanguid'].notna()], 
                       df_visit[df_visit['rocketmortgage_loanguid_evar115'].notna()], 
                       left_on = 'loanguid', 
                       right_on = 'rocketmortgage_loanguid_evar115', 
                       how = 'inner')

        print('\nTotal no. of joined records by Adobevisitorid:', df1.shape[0])
        print('\nTotal no. of joined records by RocketAccountid:', df2.shape[0])
        print('\nTotal no. of joined records by LoanNumber:', df3.shape[0])
        print('\nTotal no. of joined records by LoanGuid:', df4.shape[0])

        df = pd.concat([df1, df2, df3, df4], axis=0).drop_duplicates()
        print('\nTotal no. of joined records:', df.shape[0])
        unknown_id = set(df_visit[unique_id]) - set(df[unique_id])
        print('\nTotal no. of unjoined records with no lead form events:', len(unknown_id))
        unknown_visits = df_visit[df_visit[unique_id].isin(list(unknown_id))]
        df = pd.concat([df, unknown_visits], axis=0).reset_index(drop = True)
        
        return df


    def join_loanidentifier_with_milestone(self, df_loanidentifier, df_milestone, join='left'):

        """Join loan identifier and loan milestones on loanidentifierdimsk
        
        Parameters
        ----------
        df_loanidentifier : Pandas DataFrame
            Input loan identifier dataframe
        df_milestone : Pandas DataFrame
            Input loan milestone dataframe
        join : str
            Join method. Default to 'left'
        
        Returns
        -------
        df : Pandas DataFrame
            Return joined dataframe
        """

        df = pd.merge(df_loanidentifier, df_milestone, on = 'loanidentifierdimsk', how = join)
        df = df[df['Lead_estdatetime_min'].notna()].reset_index(drop = True)
        return df


    def join_loanstatus_with_clickstream_lead_data(self, 
                                                   df_id_lead_visit_event_agg, 
                                                   df_loanidentifier_milestone,
                                                   join_lead_created_during_visit_time=False):

        """
        Join adobe data and loan identifier milestones on loannumber and loanguid
        
        Parameters
        ----------
        df_id_lead_visit_event_agg : Pandas DataFrame
            Input adobe web data dataframe
        df_loanidentifier_milestone : Pandas DataFrame
            Input loan identifier milestone dataframe
        join_lead_created_during_visit_time : boolean
            If True, only join leads created during web visit time, i.e. lead created 
            datetime between min web visit datetime and max web visit datetime
        
        Returns
        -------
        df : Pandas DataFrame
            Return joined dataframe
        """
        df_id_lead_visit_event_agg_cols = df_id_lead_visit_event_agg.columns.to_list().copy()
        df_id_lead_visit_event_agg['min_visit_estdatetime_floormin'] = df_id_lead_visit_event_agg['min_visit_estdatetime'].dt.floor('min')
        df_id_lead_visit_event_agg['max_visit_estdatetime_ceilmin'] = df_id_lead_visit_event_agg['max_visit_estdatetime'].dt.ceil('min')
        #  match on nearest key rather than equal keys
        df1 = pd.merge_asof(df_id_lead_visit_event_agg[df_id_lead_visit_event_agg['best_match_loannumber'].\
                                                       notna()].sort_values('min_visit_estdatetime'), df_loanidentifier_milestone[[
            'Lead_estdatetime_min','loannumber','loanidentifierdimsk']].sort_values('Lead_estdatetime_min'), 
                            left_on= 'min_visit_estdatetime_floormin', 
                            right_on= 'Lead_estdatetime_min', 
                            left_by= 'best_match_loannumber', 
                            right_by= 'loannumber', 
                            direction= 'forward')
        if join_lead_created_during_visit_time:
            df1.loc[df1['Lead_estdatetime_min'] > df1['max_visit_estdatetime_ceilmin'], 'loanidentifierdimsk'] = np.NaN
        df2 = pd.merge_asof(df_id_lead_visit_event_agg[(df_id_lead_visit_event_agg['best_match_loannumber'].isna()) & (df_id_lead_visit_event_agg['best_match_loanguid'].notna())].sort_values('min_visit_estdatetime'), 
                            df_loanidentifier_milestone[['Lead_estdatetime_min','loanguid','loanidentifierdimsk']].sort_values('Lead_estdatetime_min'), 
                            left_on= 'min_visit_estdatetime_floormin', 
                            right_on= 'Lead_estdatetime_min', 
                            left_by= 'best_match_loanguid', 
                            right_by= 'loanguid', 
                            direction= 'forward')
        if join_lead_created_during_visit_time:
            df2.loc[df2['Lead_estdatetime_min'] > df2['max_visit_estdatetime_ceilmin'], 'loanidentifierdimsk'] = np.NaN
        df3 = df_id_lead_visit_event_agg[(df_id_lead_visit_event_agg['best_match_loannumber'].isna()) 
                                         & (df_id_lead_visit_event_agg['best_match_loanguid'].isna())]
        df = pd.concat([df1, df2, df3], axis=0).drop_duplicates()
        df = pd.merge(df[df_id_lead_visit_event_agg_cols + ['loanidentifierdimsk']], 
                      df_loanidentifier_milestone, 
                      on= 'loanidentifierdimsk', 
                      how= 'left') 
        df = df.rename(columns = {'loanidentifierdimsk': 'best_match_loanidentifierdimsk'})

        milestone_field_list = list(filter(re.compile('is_milestone').match, df.columns.to_list()))
        milestone_estdatetime = [[i[13:] + '_estdatetime_min' if j==0 else i[13:] + '_estdatetime_max' for j in range(2)] for i in\
                                 milestone_field_list]
        milestone_estdatetime = [item for sublist in milestone_estdatetime for item in sublist]          
        col = df_id_lead_visit_event_agg_cols + ['best_match_loanidentifierdimsk'] + milestone_field_list + milestone_estdatetime
        return df[col]


### Functions to clean web leads data
def get_bridge_table_clickstream_lead_data(df, create_bridge_table=False):

    """Group ids such as adobevisitorid, rocketaccountid, loannumber and loanguid that would represent same loan or same visitor together
    
    Parameters
    ----------
    df : Pandas DataFrame
        Input adobe visit data with joined lead ids
    
    Returns
    -------
    df : Pandas DataFrame
        Return input dataframe with additional columns 'groupidx_set', 'groupidx_len', 'groupidx_string'
    id_bridge : Pandas DataFrame
        Return id mappings. All the mapped ids are assigned with same 'groupidx_string'
    """

    identifier_cols = ['adobevisitorid', 'rocketaccountid', 'loannumber', 'loanguid', 'mcvisid_visitorid',
                      'rocketaccountid_evar5', 'rocketmortgage_loannumber_evar8', 'rocketmortgage_loanguid_evar115']
    
    rsuffix = '_group_identifier'
    group_identifier = [x + rsuffix for x in identifier_cols]
    
    print('[INFO] group data by UIDs...')
    for col in tqdm(identifier_cols):
        df = get_group_idx(df, col, rsuffix)

    print('[INFO] fill NaNs...')
    df[group_identifier] = df[group_identifier].progress_apply(lambda x: x.fillna({i: [] for i in df.index}))
    # sum across columns
    # concatenate all group identifiers into a single column
    print('[INFO] concatenate UIDs into a single column and sort...')
    df['group_idx_set'] = df[group_identifier].progress_apply(lambda x: sorted(set(x.sum())), axis=1)
    print('[INFO] determine length of indices')
    df['groupidx_len'] = df['group_idx_set'].progress_apply(lambda x: len(x))
    print('[INFO] convert UIDs to string type...')
    df['group_idx'] = df['group_idx_set'].progress_apply(str)
    
    if create_bridge_table == True:
        composite_identifier_mappings = {'unique_rocketaccountid': ['rocketaccountid', 'rocketaccountid_evar5'],
                                         'unique_loannumber': ['loannumber', 'rocketmortgage_loannumber_evar8'],
                                         'unique_loanguid': ['loanguid', 'rocketmortgage_loanguid_evar115'],
                                         'unique_adobevisitorid': ['adobevisitorid', 'mcvisid_visitorid']}
        id_table = df[identifier_cols + ['group_idx']]
        print('[INFO] create a bridge table to resolve many-to-many relationships...')
        bridge_table_id_resolution = id_table.groupby(['group_idx']).progress_apply(combine_id,
                                                                                    id_map=composite_identifier_mappings)
        return df, bridge_table_id_resolution
    else:
        return df, None


def get_matched_clickstream_lead_data(df):

    """Clean adobe visit records so that every row has unique mcvisid_visitorid + uniquevisitkey
       Obtain best match IDs where the ID created datetime is the closest to adobe visit start datetime 
    
    Parameters
    ----------
    df : Pandas DataFrame
        Input adobe visit data with joined lead ids
    
    Returns
    -------
    df : Pandas DataFrame
        Return cleaned dataframe
    """

    pd.options.mode.chained_assignment = None

    logger.info('[INFO] sorting data and resetting index...')
    df['leadsubmit_diff'] = df['leadsubmissionsent_estdatetime'] - df['min_visit_estdatetime']
    df['accountcreate_diff'] = df['accountcreate_estdatetime'] - df['min_visit_estdatetime']
    df['preloan_diff'] = df['preloan_estdatetime'] - df['min_visit_estdatetime']
    df['datetime_diff'] = df[['leadsubmit_diff','accountcreate_diff','preloan_diff']].min(axis=1)
    sort_col = ['group_idx',
                'min_visit_estdatetime',
                'datetime_diff',
                'first_homedescription',
                'first_creditrating',
                'first_timeframetopurchase',
                'first_firsttimebuyer']
    sort_order = [True] * len(sort_col)
    df = df.sort_values(by=sort_col, ascending=sort_order).reset_index(drop=True)
    
    logger.info('[INFO] aggregating records...')
    events = [f[9:] for f in df.columns if f[:8] == 'is_event']
    # window function to compute a value for each row in the groupby window/partition
    agg_dict1 = {'is_event_' + e: lambda x: max(x) for e in events}
    agg_dict2 = {e + '_estdatetime': lambda x: get_first_element(x) for e in events}
    agg_dict3 = {'visitnumber': lambda x: min(x),
                 'min_visit_estdatetime': lambda x: min(x),
                 'max_visit_estdatetime': lambda x: min(x),
                 'min_landingpage_estdatetime': lambda x: get_first_element(x),
                 'leadform_event': lambda x: max(x),
                 'min_testexperiment_estdatetime': lambda x: get_first_element(x),
                 'max_testexperiment_estdatetime': lambda x: get_first_element(x),
                 'testexperiment': lambda x: get_first_element(x),
                 'first_testexperiment': lambda x: get_first_element(x),
                 'first_marketingchannel_finder': lambda x: get_first_element(x),
                 'first_marketingchannel_closer': lambda x: get_first_element(x),
                 'first_loanpurpose': lambda x: get_first_element(x),
                 'last_loanpurpose': lambda x: get_first_element(x),
                 'first_landing_sitesection': lambda x: get_first_element(x),
                 'first_homedescription': lambda x: get_first_element(x), 
                 'first_propertyuse': lambda x: get_first_element(x),
                 'first_creditrating': lambda x: get_first_element(x), 
                 'first_timeframetopurchase': lambda x: get_first_element(x), 
                 'first_firsttimebuyer': lambda x: get_first_element(x),
                 'first_hasrealestateagent': lambda x: get_first_element(x),
                 'first_purchaseprice': lambda x: get_first_element(x),
                 'loannumber': lambda x: get_first_element(x),
                 'loanguid': lambda x: get_first_element(x),
                 'rocketaccountid': lambda x: get_first_element(x),
                 'rocketmortgage_loannumber_evar8': lambda x: get_first_element(x),
                 'rocketmortgage_loanguid_evar115': lambda x: get_first_element(x),
                 'rocketaccountid_evar5': lambda x: get_first_element(x),
                 'email': lambda x: get_first_element(x)
                 }
    agg_dict = {**agg_dict1, **agg_dict2, **agg_dict3}
    # partition data by 'group_idx', 'mcvisid_visitorid', 'uniquevisitkey'
    output = df.groupby(['group_idx', 'mcvisid_visitorid', 'uniquevisitkey'], 
                        dropna=False).aggregate(agg_dict).reset_index()
    # assign loan numbers if available through multiple sources
    logger.info('[INFO] finding valid loan number for each record...')
    output['best_match_loannumber'] = output.progress_apply(lambda x: x['loannumber'] if x['loannumber'] 
                                                   != None else x['rocketmortgage_loannumber_evar8'] if x['rocketmortgage_loannumber_evar8'] 
                                                   != None else pd.NA, axis = 1)
    # assign loanguid if available through multiple sources
    logger.info('[INFO] finding valid loanguid for each record...')
    output['best_match_loanguid'] = output.progress_apply(lambda x: x['loanguid'] if x['loanguid'] 
                                                 != None else x['rocketmortgage_loanguid_evar115'] if x['rocketmortgage_loanguid_evar115'] 
                                                 != None else pd.NA, axis = 1)
    # assign rocketaccountid if available through multiple sources
    logger.info('[INFO] finding valid rocketaccountid for each record...')
    output['best_match_rocketaccountid'] = output.progress_apply(lambda x: x['rocketaccountid'] if x['rocketaccountid'] 
                                                        != None else x['rocketaccountid_evar5'] if x['rocketaccountid_evar5'] 
                                                        != None else pd.NA, axis = 1)
    
    logger.info('[INFO] returning selected columns...')
    # select columns to return data for
    cols = ['group_idx',
            'mcvisid_visitorid',
            'email',
            'uniquevisitkey',
            'visitnumber',
            'min_visit_estdatetime',
            'max_visit_estdatetime',
            'min_landingpage_estdatetime',
            'leadform_event',
            'min_testexperiment_estdatetime',
            'max_testexperiment_estdatetime',
            'testexperiment',
            'first_testexperiment',
            'first_marketingchannel_finder',
            'first_marketingchannel_closer',
            'first_loanpurpose',
            'last_loanpurpose',
            'first_landing_sitesection',
            'first_homedescription',
            'first_propertyuse',
            'first_creditrating',
            'first_timeframetopurchase',
            'first_firsttimebuyer',
            'first_hasrealestateagent',
            'first_purchaseprice',
            'best_match_loannumber',
            'best_match_loanguid',
            'best_match_rocketaccountid']
    
    cols = cols + list(agg_dict1.keys()) + list(agg_dict2.keys())
    logger.info('[INFO] process completed!')
    
    return output[cols]

def transform_call_communication(dataFrame, post_lead_trim_comm=True, post_lead_trim_period=48):

    """
    Trim, clean, and transform call communication data. Trim allows to discard any communication
    that takes place after provided or pre-defined 'post_lead_trim_period' + lead submit timestamp
    
    Parameters
    ----------
    dataFrame : Pandas DataFrame
        Input adobe visit data with joined lead ids
    post_lead_trim_comm : boolean
        Determines if communication data is to be trimmed post lead submit event
    post_lead_trim_period : int
        Trim communication data by this period post lead submit
    
    Returns
    -------
    dataFrame : Pandas DataFrame
        Return cleaned dataframe
    """
    
    if post_lead_trim_comm == True:
        post_lead_time_elapsed = pd.to_timedelta(post_lead_trim_period, unit='h')
        dataFrame['post_lead_time_elapsed'] = dataFrame['lead_estdatetime_min'] + post_lead_time_elapsed
    else:
        dataFrame['post_lead_time_elapsed'] = dataFrame['credit_estdatetime_min']
    # substitute values where credit event occurs before post_lead_time_elapsed to prevent inclusion of calls after a credit event
    dataFrame.loc[(dataFrame['post_lead_time_elapsed'] > dataFrame['credit_estdatetime_min']), 
                  'post_lead_time_elapsed'] = dataFrame['credit_estdatetime_min']
    # One-hot encoding of column 'communication_direction'
    one_hot_encoded_cols = pd.get_dummies(dataFrame['communication_direction'])
    # Join the encoded df
    dataFrame = dataFrame.join(one_hot_encoded_cols)
    dataFrame['communication_start_date_time_ceiling'] = dataFrame['communication_start_date_time'].dt.ceil('min')
    dataFrame['communication_end_date_time_ceiling'] = dataFrame['communication_end_date_time'].dt.ceil('min')
    
    dataFrame['lead_credit_delta_est'] = dataFrame['credit_estdatetime_min'] - dataFrame['lead_estdatetime_min']
    dataFrame['lead_credit_delta_est'] = dataFrame['lead_credit_delta_est'] / np.timedelta64(1, 'h')
    dataFrame[dataFrame['is_milestone_lead'] == 1]['lead_credit_delta_est'].describe()
    # number of records with credit pull before lead submission
    negative_lead_credit_delta_id = dataFrame[(dataFrame['is_milestone_lead'] == 1) \
                                              & (dataFrame['lead_credit_delta_est'] < 0)].drop_duplicates(subset='mcvisid_visitorid')['mcvisid_visitorid'].to_list()
    dataFrame = dataFrame[~dataFrame.mcvisid_visitorid.isin(negative_lead_credit_delta_id)]
    dataFrame[dataFrame['is_milestone_lead'] == 1]['lead_credit_delta_est'].describe()
    dataFrame['call_duration'] = dataFrame['communication_end_date_time_ceiling'] - dataFrame['communication_start_date_time_ceiling']
    # define meaningful communication as duration > 60s
    #dataFrame.loc[dataFrame['call_duration'] <= pd.to_timedelta(1, unit='m'), 'call_duration'] = pd.to_timedelta(0, unit='m')
    # this may also be defined based on call status as 'connected'
    dataFrame.loc[dataFrame['communication_status'] != 'CONNECTED', 'call_duration'] = pd.to_timedelta(0, unit='m')
    # descriptive statistics of 'duration' if communication status is not CONNECTED
    dataFrame[dataFrame['communication_status'] == 'HangupOnClient']['call_duration'].describe()
    dataFrame.loc[(dataFrame['communication_start_date_time'] >= dataFrame['post_lead_time_elapsed']), 'call_duration'] = pd.to_timedelta(0, unit='m')
    dataFrame.loc[(dataFrame['communication_start_date_time'] < dataFrame['lead_estdatetime_min']), 'call_duration'] = pd.to_timedelta(0, unit='m')
    # create binary variable for client calls
    dataFrame['is_milestone_call'] = np.nan
    # Applying conditions to populate 'is_milestone_call'
    dataFrame.loc[(dataFrame['communication_start_date_time'] >= dataFrame['lead_estdatetime_min']) \
                  & (dataFrame['communication_start_date_time'] < dataFrame['post_lead_time_elapsed']), 'is_milestone_call'] = 1
    # consider all calls as attempts and subsequently identify valid attempts
    dataFrame['incoming_attempts'] = dataFrame['incoming']
    dataFrame['outgoing_attempts'] = dataFrame['outgoing']
    dataFrame.loc[(dataFrame['is_milestone_call'] != 1), 'incoming_attempts'] = 0
    dataFrame.loc[(dataFrame['is_milestone_call'] != 1), 'outgoing_attempts'] = 0
    # assign call milestone 0s if they are not considered meaningful 'CONNECTED' or are outside of bounds
    dataFrame.loc[(dataFrame['call_duration'] == pd.to_timedelta(0, unit='m')), 'is_milestone_call'] = 0
    # calls not considered meaningful 'CONNECTED' are not included here, need to refer to call attempts for that detail
    dataFrame.loc[(dataFrame['is_milestone_call'] != 1), 'incoming'] = 0
    dataFrame.loc[(dataFrame['is_milestone_call'] != 1), 'outgoing'] = 0
    # drop duplicates
    call_data_aggregated = dataFrame[['mcvisid_visitorid',
                                      'communication_start_date_time',
                                      'call_duration',
                                      'is_milestone_call',
                                      'incoming',
                                      'outgoing',
                                      'incoming_attempts',
                                      'outgoing_attempts']].drop_duplicates(['mcvisid_visitorid',                                                                                                   'communication_start_date_time']).groupby(['mcvisid_visitorid'])\
                                [['call_duration', 'is_milestone_call', 'incoming', 'outgoing', 'incoming_attempts', 'outgoing_attempts']].agg({'call_duration': 'sum','is_milestone_call': 'sum', 'incoming': 'sum', 'outgoing': 'sum', 'incoming_attempts': 'sum', 'outgoing_attempts': 'sum'}).reset_index()
    # merge to parent Pandas DataFrame
    dataFrame = dataFrame.merge(call_data_aggregated, how='left', on='mcvisid_visitorid')
    # rename columns
    dataFrame.rename(columns={'call_duration_y': 'call_duration_sum',
                              'is_milestone_call_y': 'is_milestone_call_sum',
                              'call_duration_x': 'call_duration',
                              'is_milestone_call_x': 'is_milestone_call',
                              'incoming_y': 'incoming_sum',
                              'incoming_x': 'incoming',
                              'outgoing_y': 'outgoing_sum',
                              'outgoing_x': 'outgoing',
                              'incoming_attempts_y': 'incoming_attempts_sum',
                              'incoming_attempts_x': 'incoming_attempts',
                              'outgoing_attempts_y': 'outgoing_attempts_sum',
                              'outgoing_attempts_x': 'outgoing_attempts'}, inplace=True)

    # deleting 'call_data_aggregated' to save memory
    del(call_data_aggregated)
    # create min and max for communication timestamps
    dataFrame['communication_start_date_time_min'] = dataFrame.groupby('mcvisid_visitorid')['communication_start_date_time'].transform('min')
    dataFrame['communication_start_date_time_max'] = dataFrame.groupby('mcvisid_visitorid')['communication_start_date_time'].transform('max')
    # drop duplicates
    dataFrame = dataFrame.sort_values(by=['best_match_loannumber_first', 
                                          'min_landingpage_estdatetime_first', 
                                          'communication_start_date_time'], 
                                      ascending=True, 
                                      na_position='last').drop_duplicates(subset=['mcvisid_visitorid'], keep='first')

    # calls not considered meaningful 'CONNECTED' are not included here, need to refer to call attempts for that detail
    dataFrame.loc[(dataFrame['is_milestone_call_sum'] > 0), 'is_milestone_call'] = 1
    dataFrame.loc[(dataFrame['is_milestone_call_sum'] == 0), 'is_milestone_call'] = 0
    # rename columns
    dataFrame.rename(columns={'is_milestone_call_sum': 'call_count'}, inplace=True)
    # convert call duration to seconds
    dataFrame['call_duration'] = dataFrame['call_duration'] / np.timedelta64(1, 's')
    dataFrame['call_duration_sum'] = dataFrame['call_duration_sum'] / np.timedelta64(1, 's')
    
    return dataFrame

def sql_translate(query, from_dialect, to_dialect, **kwargs):
    
    """
    This function is used to format SQL or translate between 19 different dialects like DuckDB, 
    Presto, Spark, Snowflake, and BigQuery. It aims to read a wide variety of SQL inputs and output 
    syntactically correct SQL in the targeted dialects.
    
    Parameters
    ----------
    query : str
        Input query that needs to be translated
    from_dialect : str
        Source sql instruction dialect
    to_dialect : str
        Destination sql instruction dialect
    
    Returns
    -------
    query : str
        Return translated sql query
    """
    
    for key, value in kwargs.items():
        if type(value) == str:
            value = '"' + value + '"'
        else:
            value = str(value)
        query = query.replace(str(key), value)
        query = query.replace(':', '')
        query = query.replace(';', '')
    query = sqlglot.transpile(query, read=from_dialect, write=to_dialect, identify=True, pretty=True)[0]
    
    return query