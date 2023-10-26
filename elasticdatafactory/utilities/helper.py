# import sys functions and variables to manipulate Python runtime environment
import sys
# import AWS SDK for Python
import boto3
# import awswrangler package
import awswrangler as wr
# import pandas for relational data analysis and manipulation
import pandas as pd
# import pathlib
from pathlib import Path
# import inspect to populate class objects
import inspect
# import pathlib
from pathlib import Path
# import logger
import logging
# import elastic data factory query wrapper
from elasticdatafactory.data import make_dataset
# import utility functions
from elasticdatafactory.utilities import utility

# load logger
logger = logging.getLogger('')

"""
This module provides helper functions that are used within Data Factory, 
and these can be useful for external consumption as well.
"""

class FeatureMapping():
    
    """A class used to generate field and query mappings

    Attributes
    ----------
    query_registry : str
        Name of the module containing query classes
    query_db_map_loc : str
        Location of JSON file with database and query mappings to be loaded as an input
    field_query_map_loc : str
        Location of JSON file to be saved with field and query mappings
   
    """
    
    def __init__(self, query_registry, query_db_map_loc, field_query_map_loc):   
        
        self.query_registry = query_registry
        self.query_db_mapping = utility.import_json(query_db_map_loc)
        self.field_query_map_loc = field_query_map_loc
        
    def create_default_args(self, start_date, end_date, query_label, database, table_name, limit=1, workgroup='rcd-datascientist', output_filepath=None):
        
        """A function used to generate basic query parameters

        Attributes
        ----------
        self : object
            FeatureMapping class instance
        start_date : str
            Start date for the query in 'yyyy-mm-dd' format 
        end_date : str
            End date for the query in 'yyyy-mm-dd' format
        query_label : str
            Name of the query class, for instance 'QueryExperimentVisitorData'
        database : str
            Database name for retrieving tabulated data, for instance 'rktdp_adobe_omniture_raw_processed_access'
        table_name : str
            Table name for retrieving data, for instance 'adobe_dq_prcd_data'
        limit : int
            Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified     
        
        Returns
        -------
        args : dict
            Return query arguments as dictionary
        """

        args = {'start_date': start_date, 
            'end_date': end_date, 
            'query_label': query_label, 
            'database': database, 
            'table_name': table_name,
            'limit': limit,
            'workgroup': workgroup,
            'output_filepath': output_filepath
           }
        return args
    
    def extract_fields(self, object_name):
        
        """A functiont to extract fields from each table in a given database.

        Parameters
        ----------
        self : object
            FeatureMapping class instance
        object_name : str
            Name of the query class, for instance 'QueryExperimentVisitorData'
        
        Returns
        -------
        List : list
            Return list of fields in each queried table
        
        """

        if object_name != 'QueryDataBase':
            args = self.create_default_args(start_date='2022-06-21', 
                                            end_date='2022-06-22', 
                                            query_label=object_name, 
                                            database=self.query_db_mapping[object_name][0], 
                                            table_name=self.query_db_mapping[object_name][1])
            dataFrame = make_dataset.main.callback(**args)
            if dataFrame is not None:
                return list(dataFrame.columns.values)
            else:
                return []
        
        
    def generate_mapping(self):
        
        """A function to generate mappings of fields in tables to corresponding query classes

        Parameters
        ----------
        self : object
            FeatureMapping class instance
        
        Returns
        -------
        field_query_mapping : dict
            Return mapped dictionary

        """
        
        field_query_mapping = dict()
        object_dict = utility.populate_modules(module_name=self.query_registry)
        for object_name, obj in object_dict.items():
            try:
                field_query_mapping[object_name] = self.extract_fields(object_name)
            except ValueError:
                continue
        # export json file
        utility.export_json(self.field_query_map_loc, field_query_mapping)
        return field_query_mapping
    
    
    
### Functions to get final output dataset
T = utility.Transformation()
J = utility.Join()
required_args = {'limit': 'ALL',  'workgroup': 'rcd-datascientist', 'output_filepath': None}

def get_lead_ids(start_date, end_date, loan_purpose='', required_args=required_args):
    
    """Queries lead created data with date range
    
    Parameters
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    loan_purpose : str
        Filter output with loan purpose 'Refinance' or 'Purchase'. Default to ''

    Returns
    -------
    df_id : pd.DataFrame
        Return pandas dataframe
    """
    df_preloan_id = make_dataset.main.callback(**{**required_args,
                                                  **{'query_label': 'QueryPreloanId', 
                                                     'start_date': start_date, 
                                                     'end_date': end_date, 
                                                     'database': "rm_dp_qtweets_raw_processed_access"}
                                                 })
    print('\nNo. of Preloan ID records:', df_preloan_id.shape[0])
    df_preloan_purpose = make_dataset.main.callback(**{**required_args,
                                                       **{'query_label': 'QueryPreloanPurpose', 
                                                          'start_date': start_date, 
                                                          'end_date': end_date, 
                                                          'loan_purpose': loan_purpose, 
                                                          'database': "rm_dp_qtweets_raw_processed_access"}
                                                      })
    print('\nNo. of Preloan purpose records:', df_preloan_purpose.shape[0])
    df_account = make_dataset.main.callback(**{**required_args,
                                               **{'query_label': 'QueryAccountCreate',
                                                  'start_date': start_date, 
                                                  'end_date': end_date,  
                                                  'database': "rm_dp_qtweets_raw_processed_access"}
                                              })
    print('\nNo. of Account create records:', df_account.shape[0])
    df_lead = make_dataset.main.callback(**{**required_args,
                                            **{'query_label': 'QueryLeadSubmission',
                                            'start_date': start_date, 
                                            'end_date': end_date, 
                                            'loan_purpose': loan_purpose, 
                                            'database': "rm_dp_submissionengine_raw_processed_access"}
                                           })
    print('\nNo. of Lead submit records:', df_lead.shape[0])

    df_preloan = J.join_preloan_id_with_purpose(df_preloan_id, df_preloan_purpose)
    df_account = T.transform_account_df(df_account)
    df_lead = T.transform_lead_df(df_lead)
    df_id = J.join_lead_with_preloan_with_account(df_lead, df_preloan, df_account)
    print('\nNo. of joined id records:', df_id.shape[0])
    return df_id


def get_adobe_visit(start_date, end_date, loan_purpose='', group_id='', site_sections=['ql lander', 'rocket lander'], event_codes=['204', '206', '250'], required_args=required_args):
    
    """Queries adobe visit data with date range
    
    Parameters
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    loan_purpose : str
        Filter output with loan purpose 'Refinance' or 'Purchase'. Default to ''
    group_id : str
        ID for experiment group, for instance '562585:0:0'. Default to ''
    site_sections : list
        Filter data with selected sitesections. Default to ['ql lander', 'rocket lander']
    event_data : list
        Filter data with a list of adobe event code. Default to ['204', '206', '250']

    Returns
    -------
    df_lead_visit_event : pd.DataFrame
        Return pandas dataframe
    """
    
    df_event = make_dataset.main.callback(**{**required_args,
                                             **{'query_label': 'QueryAdobeEvents',
                                                'start_date': start_date, 
                                                'end_date': end_date, 
                                                'event_codes': event_codes, 
                                                'database': "rktdp_adobe_omniture_raw_processed_access"}
                                            })
    print('\nNo. of event records:', df_event.shape[0])
    df_leadform_input = make_dataset.main.callback(**{**required_args,
                                                      **{'query_label': 'QueryAdobeLeadformInput',
                                                         'start_date': start_date, 
                                                         'end_date': end_date, 
                                                         'loan_purpose': loan_purpose,  
                                                         'group_id': group_id, 
                                                         'site_sections': site_sections,
                                                         'database': "rktdp_adobe_omniture_raw_processed_access"}
                                                     })
    print('\nNo. of leadform input records:', df_leadform_input.shape[0])
    
    df_event = T.transform_event_df(df_event)
    df_leadform_input = T.transform_leadform_input_df(df_leadform_input)
    df_lead_visit_event = J.join_event_visitor_with_leadform_input(df_event, df_leadform_input)
    return df_lead_visit_event


def get_loanmilestone_status(start_date, end_date, milestones=['Lead', 'Credit', 'PAL', 'VAL'], long_to_wide_column='loanmilestone_groupname', required_args=required_args):
    
    """Queries adobe visit data with date range
    
    Parameters
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    milestones : list
        Filter data with selected milestones. Default to ['Lead', 'Credit', 'PAL', 'VAL']
    long_to_wide_column : str
        Select long_to_wide_column to transform loan milestone dataset from long to wide. Select 'loanmilestone_groupname' or 'loanmilestone_eventname' for this field. Default to 'loanmilestone_groupname'

    Returns
    -------
    df_loanidentifier_milestone : pd.DataFrame
        Return pandas dataframe
    """
    
    df_loanidentifier = make_dataset.main.callback(**{**required_args,
                                                      **{'query_label': 'QueryLoanIdentifier',
                                                         'start_date': start_date, 
                                                         'end_date': end_date, 
                                                         'database': "rm_northstar_raw_processed_access"}
                                                     })
    print('\nNo. of loanidentifier records:', df_loanidentifier.shape[0])
    df_milestone = make_dataset.main.callback(**{**required_args,
                                                 **{'query_label': 'QueryMajorMilestone',
                                                    'start_date': start_date, 
                                                    'end_date': end_date,  
                                                    'milestones': milestones,
                                                    'database': "rm_northstar_raw_processed_access"}
                                                })
    print('\nNo. of loan milestone records:', df_milestone.shape[0])
    
    df_loanidentifier = T.transform_loanidentifier_df(df_loanidentifier)
    df_milestone = T.transform_milestone_df(df_milestone, long_to_wide=True, long_to_wide_column=long_to_wide_column)
    df_loanidentifier_milestone = J.join_loanidentifier_with_milestone(df_loanidentifier, df_milestone)
    return df_loanidentifier_milestone
    
    
def get_joined_dataset(start_date, end_date, loan_purpose='', group_id='', site_sections=['ql lander', 'rocket lander'], event_codes=['204', '206', '250'], milestones=['Lead', 'Credit', 'PAL', 'VAL'], milestone_long_to_wide_column='loanmilestone_groupname', join_lead_created_during_visit_time=True):
    
    """Queries adobe visit data with date range
    
    Parameters
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    loan_purpose : str
        Filter output with loan purpose 'Refinance' or 'Purchase'. Default to ''
    group_id : str
        ID for experiment group, for instance '562585:0:0'. Default to ''
    site_sections : list
        Filter data with selected sitesections. Default to ['ql lander', 'rocket lander']
    event_data : list
        Filter data with a list of adobe event code. Default to ['204', '206', '250']
    milestones : list
        Filter data with selected milestones. Default to ['Lead', 'Credit', 'PAL', 'VAL']
    long_to_wide_column : str
        Select long_to_wide_column to transform loan milestone dataset from long to wide. Select 'loanmilestone_groupname' or 'loanmilestone_eventname' for this field. Default to 'loanmilestone_groupname'
    join_lead_created_during_visit_time : boolean
        If True, only join leads created during web visit time, i.e. lead created datetime between min web visit datetime and max web visit datetime. Default to True

    Returns
    -------
    dff : pd.DataFrame
        Return pandas dataframe
    id_bridge : pd.DataFrame
        Return id mappings. All the mapped ids are assigned with same 'groupidx_string'
    """
    
    print('Query data for date between {} and {}'.format(start_date, end_date))
    
    print('\nGet lead ids')
    df_id = get_lead_ids(start_date, end_date, loan_purpose)
    
    print('\nGet adobe visit data')
    df_lead_visit_event = get_adobe_visit(start_date, end_date, loan_purpose, group_id, site_sections, event_codes)
    df_web_data = J.join_lead_id_with_clickstream_lead_data(df_id, df_lead_visit_event, unique_id='uniquevisitkey')

    df_web_data, id_bridge = utility.get_bridge_table_clickstream_lead_data(df_web_data)
    df_web_data = utility.get_matched_clickstream_lead_data(df_web_data)
    print('\nNo. of web data records after mapping ids:', df_web_data.shape[0])
          
    print('\nGet loan milestone data')
    df_loanidentifier_milestone = get_loanmilestone_status(start_date, end_date, milestones, milestone_long_to_wide_column)
    
    dff = J.join_loanstatus_with_clickstream_lead_data(df_web_data, df_loanidentifier_milestone, join_lead_created_during_visit_time)   
    print('\nNo. of final joined records:', dff.shape[0])
    return dff, id_bridge
    

    
if __name__ == '__main__':
    
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
