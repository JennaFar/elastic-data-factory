# -*- coding: utf-8 -*-
# importing necessary libraries to data processing
# import functions for interacting with the operating system
import os
# import sys functions and variables to manipulate Python runtime environment
import sys
# import AWS SDK for Python
import boto3
# import AWS SDK for pandas to integrate dataframes with several AWS Services
import awswrangler as wr
# import pandas for relational data analysis and manipulation
import pandas as pd
# import date module to get current date
from datetime import date
# import click for creating command line interfaces
import click
# import pathlib to handle and manipulate file path
from pathlib import Path
# import importlib for dynamic imports during runtime
import importlib
# import custom logger for logging purposes
from elasticdatafactory import setup_custom_logger

# initiate persistent session
session = boto3.Session()
# specify s3 client
s3 = boto3.client('s3')
wr.config.sts_endpoint_url = "https://sts." + session.region_name + ".amazonaws.com"
# provide env variables
os.environ['AWS_DEFAULT_REGION'] = 'us-east-2' # specify your AWS region.
# import logger
logger = setup_custom_logger('')

class QueryWrapper():
    
    """A query wrapper to offer a convenient approach to execute pre-defined queries
    
    Attributes
    ----------
    query : str
        A string representing data query
    params : dict
        Input params for parameterized data query
    database : str
        A string representing database to query data from
    workgroup : str
        A string representing Work Group name
    ctas_approach : boolean
        A data object for CREATE TABLE AS SELECT (CTAS) option that creates a new table in Athena from the results of a SELECT statement from another query
        
    """  
    def __init__(self, query, params, database, workgroup, ctas_approach=False, **kwargs):
        
        self.query = query
        self.params = params
        self.database = database
        self.workgroup = workgroup
        self.ctas_approach = ctas_approach

    def execute_query(self):
        
        """Queries data based on provided parameters.
        
        Parameters
        ----------
        self : object
            QueryWrapper class instance
        
        Returns
        -------
        queried_data : pd.DataFrame
            Return the results of the query as DataFrame
            
        """

        if self.params is not None:
            queried_data = wr.athena.read_sql_query(sql=self.query, 
                                                    database=self.database, 
                                                    params=self.params, 
                                                    ctas_approach=self.ctas_approach, 
                                                    workgroup=self.workgroup,
                                                    boto3_session=session)
            return queried_data
        else:
            return None

def save(df_write, output_filepath):
    
    """Export queried data to s3 bucket as a comma-separated values file.
    
    Parameters
    ----------
    df_write : DataFrame
        Results of the query as DataFrame for .csv storage
    output_filepath : str
        Path to s3 bucket to store data

    Returns
    -------
    None

    """

    # Write dataframe to s3
    file_format = output_filepath.split('.')[-1]
    if file_format == 'csv':
        # incremental upload to an already created bucket
        df_write.to_csv(output_filepath, index=False)
    elif file_format == 'xlsx':
        df_write.to_excel(output_filepath, index=False)
    elif file_format == 'parquet':
        wr.s3.to_parquet(df_write, output_filepath)
        
def check_jupyter() -> bool:
    
    """Function to check if IDE is Jupyter notebook, terminal, 
    other unrecognized IDE's, or a standard Python interpreter.
    
    Parameters
    ----------
    None

    Returns
    -------
    boolean : True or False indicating availability of Jupyter Notebook

    """
    
    try:
        shell = get_ipython().__class__.__name__
        # Jupyter notebook or qtconsole
        if shell == 'ZMQInteractiveShell':
            return True
        # Terminal running IPython
        elif shell == 'TerminalInteractiveShell':
            return False
        # everything else
        else:
            return False
    except NameError:
        # standard Python interpreter
        return False
    
def create_new_cell(contents):
    
    """Creates a cell in Jupyter Notebook with a pre-determined payload as its contents.
    
    Parameters
    ----------
    contents : str
        Represents payload to transfer to a new cell created by this function

    Returns
    -------
    None

    """
    
    try:
        shell = get_ipython()
        payload = dict(
            source='set_next_input',
            text=contents,
            replace=False,
        )
        shell.payload_manager.write_payload(payload, single=False)
    except NameError:
        pass
    
def get_callback_function(query_label: str, database: str) -> str:
    
    """Callback function to be passed into Jupyter cell for execution.
    
    Parameters
    ----------
    query_label : str
        Name class object containing query for execution
    database : str
        Name of database to retrieve data table from
    
    Returns
    -------
    content : str
        Represents a raw string to be passed into Jupyter cell

    """
    
    content = """dataFrame = make_dataset.main.callback(**
                                        {**required_args,
                                            **{'query_label': '%s',
                                                'start_date': start_date,
                                                'end_date': end_date,
                                                'database': '%s',
                                                'edit_mode': True,
                                                'custom_query': query
                                                }
                                            }
                                        )""" %(query_label, database)
    return content

    
@click.command('main')
@click.option('--start_date', prompt='Start Date', help='Start date for the experiment in yyyy-mm-dd', type=str)
@click.option('--end_date', prompt='End Date', help='End date for the experiment in yyyy-mm-dd', type=str)
@click.option('--query_label', prompt='Query Label', help='Label representing specific query for execution', type=str)
@click.option('--database', prompt='Database', help='Name of database to retrieve data table from', type=str)
@click.option('--table_name', prompt='Table Name', is_flag=False, flag_value='None', default='default', 
              help='Table name for retrieving data', type=str, required=False)
@click.option('--group_id', prompt='Experiment Group ID', is_flag=False, flag_value='562585:1:0', default='562585:1:0', 
              help='ID for experiment group', type=str, required=False)
@click.option('--event_codes', prompt='Adobe Event Codes', is_flag=False, flag_value=[], default=[], 
              help='Filter data by a list of adobe event code', type=str, multiple=True, required=False)
@click.option('--first_only', prompt='First Occurrence of Event Only', is_flag=False, flag_value=True, default=True, 
              help='Return data for first occured event for each event in each unique visit', type=bool, required=False)
@click.option('--site_sections', prompt='Site Sections', is_flag=False, flag_value=['ql lander', 'rocket lander'], default=['ql lander', 'rocket lander'], 
              help='Filter data by selected sitesections such as ql lander or/and rocket lander', type=str, multiple=True, required=False)
@click.option('--loan_purpose', prompt='Loan Purpose', is_flag=False, flag_value='', default='', 
              help='Filter data by loan purpose such as Refinance or/and Purchase', type=str, required=False)
@click.option('--milestones', prompt='Milestones', is_flag=False, flag_value=['Lead','Net Leads','Allocated','Credit','PAL','VAL','Application','Folder','Closing'], 
              default=['Lead','Net Leads','Allocated','Credit','PAL','VAL','Application','Folder','Closing'], 
              help='Filter data by milestones such as Lead or/and PAL', type=str, multiple=True, required=False)
@click.option('--lead_type', prompt='Lead Type', is_flag=False, flag_value='website', default='website', 
              help='Filter data by lead types such as website', type=str, required=False)
@click.option('--workgroup', prompt='Workgroup', is_flag=False, flag_value='rcd-datascientist', default='rcd-datascientist', 
              help='Specify workgroup', type=str, required=False)
@click.option('--limit', prompt='Maximum Number of Records', is_flag=False, flag_value='ALL', default='ALL', 
              help='Select a maximum number of records to retrieve', required=False)
@click.argument('output_filepath', type=click.File('wb'), required=False)


def main(**kwargs): 

    """Runs data processing scripts to turn raw data from (../raw) into
       cleaned data ready to be analyzed (saved in ../processed).
       
    Parameters
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    query_label : str
        Name class object containing query for execution
    group_id : str
        ID for experiment group, for instance '562585:0:0'
    table_name : str
        Table name for retrieving experiment data, for instance 'rktdp_adobe_omniture_raw_processed_access.adobe_dq_prcd_data'
    database : str
        Name of database to retrieve data table from
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    edit_mode : boolean
        Mode that displays and allows editing of sql statement before being sent to AWS SDK for pandas (awswrangler) for execution
    custom_query : str
        Parameterized SQL query edited by user for execution
    Returns
    -------
    dataFrame : pd.DataFrame
        Returns a pandas DataFrame with results from executed SQL instruction. 
        Parameterized SQL query will be printed if edit_mode is True and Jupyter Notebook is absent
  
    """

    # specify arguments in dict() data type
    kwargs = {k: list(v) if type(v) is tuple else v for k, v in kwargs.items()}
    # apply conditions to 'limit' argument for compatibility with AWS SDK for pandas
    kwargs['limit'] = kwargs['limit'] if kwargs['limit'] == 'ALL' else int(kwargs['limit'])
    # check and impute values for 'edit_mode' key for editing SQL instructions
    kwargs['edit_mode'] = kwargs['edit_mode'] if 'edit_mode' in kwargs else False
    # check and impute values for 'custom_query' key for custom SQL instruction
    kwargs['custom_query'] = kwargs['custom_query'] if 'custom_query' in kwargs else None
    # import module and user-defined query
    query_module = importlib.import_module('elasticdatafactory.data.query_registry')
    # get Python Object based on 'query_label' class in dynamically imported module
    query_object = getattr(query_module, kwargs['query_label'])
    try:
        dataFrame = get_data(query_object(**{k: v for k, v in kwargs.items() if v != 'default'}), 
                          query_label=kwargs['query_label'],
                          database=kwargs['database'], 
                          workgroup=kwargs['workgroup'], 
                          edit_mode=kwargs['edit_mode'],
                          custom_query=kwargs['custom_query']
                         )
        if kwargs['edit_mode'] == False:
            logger.debug('[DEBUG] snapshot of acquired data \n %s', dataFrame.head(2))
            if kwargs['output_filepath'] is not None:
                save(dataFrame, kwargs['output_filepath'])
                return dataFrame
            else:
                return dataFrame
        else:
            logger.debug('[DEBUG] empty Pandas DataFrame is returned!')
            return dataFrame
    except Exception as e:
        logger.exception(e)


def get_data(query_object, query_label, database, workgroup='rcd-datascientist', edit_mode=False, custom_query=None):
    
    """Queries data based on provided parameters.
    
    Parameters
    ----------
    query_object : object
        Query class object
    query_label : str
        Name class object containing query for execution
    database : str
        Database name like 'rm_dp_qtweets_raw_processed_access'
    workgroup : str
        A string representing Work Group name
    edit_mode : boolean
        Mode that displays and allows editing of sql statement before being sent to AWS SDK for pandas (awswrangler) for execution, defaults to False
    custom_query : str
        Parameterized SQL query edited by user for execution, defaults to None
    Returns
    -------
    dataFrame : pd.DataFrame
        Return the results of the query as pandas DataFrame  
    """
    
    get_query = query_object
    query = get_query.sql_instruction()
    params = get_query.query_params()
    # call wrapper and pass built-in sql instruction with parameters
    if edit_mode == False:
        logger.info('[INFO] data acquisition in progress...')
        execute_command = QueryWrapper(query=query, 
                                       params=params, 
                                       database=database, 
                                       workgroup=workgroup
                                      )
        dataFrame = execute_command.execute_query()
        logger.debug('[DEBUG] Pandas DataFrame is returned from execution of built-in SQL instructions')
        return dataFrame
    # call wrapper and pass custom sql instruction with parameters
    elif edit_mode == True and custom_query is not None:
        logger.info('[INFO] module executed a user-defined SQL instruction')
        logger.info('[INFO] data acquisition in progress...')
        execute_command = QueryWrapper(query=custom_query, 
                                       params=params, 
                                       database=database, 
                                       workgroup=workgroup
                                      )
        dataFrame = execute_command.execute_query()
        logger.debug('[DEBUG] Pandas DataFrame is returned from execution of custom SQL instructions')
        return dataFrame
    # call callback function and pass built-in sql instruction as payload to Jupyter notebook
    elif edit_mode == True and custom_query is None:
        
        logger.info('[INFO] detecting IDE to create appropriate editing mode')
        if check_jupyter() == True:
            # display SQL instruction and callback function for editing
            create_new_cell('query = ' + '"""' + query + '"""' + '\n\n' \
                            + '# callback function to execute above SQL instruction' + '\n' \
                            + get_callback_function(query_label, database))
            logger.info('[INFO] module executed in query edit mode')
        if check_jupyter() == False:
            # return contents as is due to unavailability of Jupyter IDE
            payload = 'query = ' + '"""' + query + '"""' + '\n\n' \
            + '# callback function to execute above SQL instruction' + '\n' \
            + get_callback_function(query_label, database)
            logger.info('[WARNING] module executed in query edit mode with limited capabilities, copy/paste following raw string for execution.')
            logger.debug('[DEBUG] SQL instructions are printed with callback function to execute through wrapper')
            print('\n' + payload)
        logger.debug('[DEBUG] empty Pandas DataFrame is returned!')
        return pd.DataFrame()
    else:
        logger.error('[ERROR] module can only be executed in valid query edit modes such as True or False!')

        
if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
