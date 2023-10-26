# -*- coding: utf-8 -*-
# import AWS SDK for pandas to integrate dataframes with several AWS Services
import awswrangler as wr
# import date module to get current date
from datetime import date, datetime
# import logger for logs
import logging

# load logger
logger = logging.getLogger('')

class QueryDataBase():
    
    """A class used to generate basic query parameters

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data, for instance 'adobe_dq_prcd_data'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified     
    """
    
    def __init__(self, start_date, end_date, table_name, limit='ALL', **kwargs):   
        
        self.start_date, self.end_date = self._is_valid_date_range(start_date, end_date)
        self.table_name = self._is_valid_table_name(table_name)
        self.limit = limit
        
    def _is_valid_date_range(self, start_date, end_date):
        
        """Check if input dates are valid. Raise Value Error when dates are not in expected format. Raise Exception when end date is before start date.

        Parameters
        ----------
        self : object
            QueryDataBase class instance
        start_date : str
            Start date to query data
        ent_date
            End date to query data
        
        Returns
        -------
        start_date : str
            Return valid start date
        ent_date
            Return valid end date
        """
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        if (end.date() < start.date()):
            raise Exception("End date cannot be earlier than start date. To query one day, set end date same as start date.")
        return start_date, end_date
       
    def _is_valid_table_name(self, table_name):
        
        """Check if input table_name is valid.

        Parameters
        ----------
        self : object
            QueryDataBase class instance
        table_name : str
            Table name to query data
        
        Returns
        -------
        table_name : str
            Return table name when it is not empty string
        """
        
        if table_name == '' or type(table_name) != str:
            raise Exception("Table name cannot be empty")
        return table_name
        
    def query_params(self):
        
        """Creates parameter for data query.

        Parameters
        ----------
        self : object
            QueryDataBase class instance
        
        Returns
        -------
        params : dict
            Return query parameters as dictionary
        """
        
        params = {
        "start_date": F"'{self.start_date}'",
        "end_date": F"'{self.end_date}'",
        "table_name": F"{self.table_name}",
        "limit": F"{self.limit}"
        }
        return params
    
    

class QueryExperimentVisitorData(QueryDataBase):
    
    """A class used to represent visitor query

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'adobe_dq_prcd_data'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified   
    group_id : str
        ID for experiment group, for instance '562585:0:0'
    """    
    
    def __init__(self, start_date, end_date, table_name='adobe_dq_prcd_data', limit='ALL', group_id='562585:1:0', **kwargs):

        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """  
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
        self.group_id = group_id

    def query_params(self):
        
        """Creates parameter for data query.

        Parameters
        ----------
        self : object
            QueryVisitorData class instance
        
        Returns
        -------
        params : dict
            Return query parameters as dictionary
        """
        
        params = {
            "start_date": F"'{self.start_date}'",
            "end_date": F"'{self.end_date}'",
            "group_value_1": F"'{self.group_id}'",
            "group_value_2": F"'%--{self.group_id}%'",
            "group_value_3": F"'{self.group_id}%'",
            "table_name": F"{self.table_name}",
            "limit": F"{self.limit}",
        }
        return params

    def sql_instruction(self):
        
        """Queries data based on provided parameters.
        
        Parameters
        ----------
        self : object
            QueryVisitorData class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution
        """
        
        query = """
            WITH groupData AS /*+label(query_visitor_data)*/
                (
            SELECT 
               mcvisid_visitorid,
               LOWER(post_mvvar3) as post_mvvar3,
               post_tnt,
               datekey,
               hour,
               visitnumber,
               visit_page_num,
               pagename,
            ROW_NUMBER() OVER 
            (
            PARTITION BY 
                mcvisid_visitorid
            ORDER BY 
                datekey DESC, 
                hour DESC, 
                visitnumber DESC, 
                visit_page_num DESC
            ) 
            AS rowNumber
            FROM :table_name;
            WHERE 
                (
                post_tnt= :group_value_1; 
                OR post_tnt LIKE :group_value_2; 
                OR post_tnt LIKE :group_value_3;) 
                AND datekey >= :start_date; 
                AND datekey <= :end_date;
                AND post_mvvar3 <> ''
                ),
            distinctVisitors AS
            (
            SELECT DISTINCT
               mcvisid_visitorid,
               post_tnt,
               datekey,
               hour
            FROM :table_name;
            WHERE 
                (
                post_tnt= :group_value_1; 
                OR post_tnt LIKE :group_value_2; 
                OR post_tnt LIKE :group_value_3;) 
                AND datekey >= :start_date;
                AND datekey <= :end_date;
                )
            SELECT 
               A.mcvisid_visitorid,
               A.post_mvvar3,
               A.post_tnt,
               A.datekey,
               A.hour,
               A.visitnumber,
               A.visit_page_num,
               A.pagename,
               A.rowNumber
            FROM groupData AS A
            INNER JOIN distinctVisitors ON
                A.mcvisid_visitorid = distinctVisitors.mcvisid_visitorid
            GROUP BY 
                A.mcvisid_visitorid,
                A.post_mvvar3,
                A.post_tnt,
                A.datekey,
                A.hour,
                A.visitnumber,
                A.visit_page_num,
                A.pagename,
                A.rowNumber
            LIMIT :limit;
            """
        return query


class QueryPreloanPurpose(QueryDataBase):
    
    """A child class which inherits all methods and properties from QueryDataBase used to represent preloan purpose query

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'preloanguidcreateevent_base'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    loan_purpose : str
        Filter data with loan purpose 'Refinance' or 'Purchase'
    """
    
    def __init__(self, start_date, end_date, table_name='preloanguidcreateevent_base', limit='ALL', loan_purpose='', **kwargs):
        
        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """   
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
        self.loan_purpose = loan_purpose
       
    def sql_instruction(self):
        
        """Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryPreloanPurpose class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: datetime(UTC), recordid, loanpurpose, datekey
        """
        
        query = """SELECT datetime AS utcdatetime,
                          recordid,
                          loanpurpose,
                          datekey
                   FROM :table_name;
                   WHERE datekey >= :start_date; 
                   AND datekey <= :end_date;
                   """
        if self.loan_purpose:
            query += """ AND loanpurpose = '{}' """.format(self.loan_purpose)
        query += """ LIMIT :limit; """
        return query


class QueryPreloanId(QueryDataBase):
    
    """A child class which inherits all methods and properties from QueryDataBase used to represent preloan id query

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'preloanguidcreateevent_sourcereferenceidentifiers_sourcereferenceidentifier'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    """ 
    
    def __init__(self, start_date, end_date, table_name='preloanguidcreateevent_sourcereferenceidentifiers_sourcereferenceidentifier', limit='ALL', **kwargs):
        
        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """ 
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
       
    def sql_instruction(self):
        
        """Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryPreloanId class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: recordid, loanguid, loannumber, rocketaccountid, datekey
        """
        
        query = """WITH AppStart AS
                   (
                   SELECT recordid,
                          sourcereferenceidentifierssourcereferenceidentifiervalue AS loanguid,
                          datekey
                   FROM :table_name;
                   WHERE datekey >= :start_date;
                   AND datekey <= :end_date;
                   AND sourcereferenceidentifierssourcereferenceidentifiersource = 'RocketMortgage'
                   AND sourcereferenceidentifierssourcereferenceidentifiertype = 'PreLoan'
                   ),
                   Loan AS 
                   (
                   SELECT recordid,
                          sourcereferenceidentifierssourcereferenceidentifiervalue AS loannumber
                   FROM :table_name;
                   WHERE datekey >= :start_date;
                   AND datekey <= :end_date;
                   AND sourcereferenceidentifierssourcereferenceidentifiertype = 'Loan'
                   ),
                   Account AS 
                   (
                   SELECT recordid,
                          sourcereferenceidentifierssourcereferenceidentifiervalue AS rocketaccountid
                   FROM :table_name;
                   WHERE datekey >= :start_date;
                   AND datekey <= :end_date;
                   AND sourcereferenceidentifierssourcereferenceidentifiersource = 'RocketAccount'
                   )
                   SELECT S.recordid,
                          S.loanguid,
                          L.loannumber,
                          A.rocketaccountid,
                          S.datekey
                   FROM AppStart S
                   LEFT JOIN Loan L ON 
                       L.recordid = S.recordid
                   LEFT JOIN Account A ON
                       A.recordid = S.recordid
                   LIMIT :limit;
                   """
        return query


class QueryAccountCreate(QueryDataBase):
    
    """A child class which inherits all methods and properties from QueryDataBase used to represent account create query

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'accountactivityevent_base'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    """
    
    def __init__(self, start_date, end_date, table_name='accountactivityevent_base', limit='ALL', **kwargs):
        
        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """ 
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
       
    def sql_instruction(self):
        
        """Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryAccountCreate class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: datetime(UTC), recordid, rocketaccountid, datekey
        """
        
        query = """SELECT DISTINCT 
                       datetime AS utcdatetime,
                       recordid,
                       sourcereferenceidentifierssourcereferenceidentifiervalue AS rocketaccountid,
                       datekey
                   FROM :table_name;
                   WHERE datekey >= :start_date;
                   AND datekey <= :end_date;
                   AND actiontype = 'Account Create'
                   LIMIT :limit;
                   """
        return query


class QueryLeadSubmission(QueryDataBase):
    
    """A child class which inherits all methods and properties from QueryDataBase used to represent lead submission query

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'submissionstrings_base'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    lead_type : str
        Filter data with specific lead type. By default, this is set to 'website'
    loan_purpose : str
        Filter data with loan purpose 'Refinance' or 'Purchase'
    """
    
    def __init__(self, start_date, end_date, table_name='submissionstrings_base', limit='ALL', lead_type='website', loan_purpose='', **kwargs):
        
        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """   
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
        self.lead_type = lead_type 
        self.loan_purpose = loan_purpose
       
    def sql_instruction(self):
        
        """Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryLeadSubmission class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. 
            Primary fields include: 
                datetime(EST), 
                datetime(UTC), 
                recordid, 
                leadid, 
                loannumber, 
                adobevisitorid, 
                loanguid, 
                rmclientid, 
                gcid, 
                gcid2, 
                loanpuporse, 
                leadsource, 
                datekey
        """
        
        query ="""WITH T AS
                  (
                  SELECT DISTINCT 
                       timestamp AS estdatetime,
                       payloadutcdatetimeleadreceived AS leadreceived_utcdatetime,
                       recordid,
                       payloadhtleadid AS leadid,
                       payloadloannumber AS loannumber,
                       payloadadobevisitorid AS adobevisitorid,
                       payloadrmloanid AS loanguid,
                       payloadrmclientid AS rmclientid,
                       payloaduniversalloanidentifier AS loanidentifier,
                       CASE WHEN LOWER(payloadhsloan) IN ('refinance','refi')
                            THEN 'Refinance'
                            WHEN LOWER(payloadhsloan) = 'purchase'
                            THEN 'Purchase'
                            ELSE payloadhsloan
                       END AS loanpurpose,
                       payloadhtleadsystem AS leadsystem,
                       payloadleadsource AS leadsource,
                       payloadleadsourcecategory AS leadsourcecategory,
                       payloadleadtypecode AS leadtypecode,
                       payloadtypeoflead AS typeoflead,
                       payloadupemgrade AS loangrade,
                       payloadhtisduplicatelead AS isduplicate,
                       payloadhsrlautotransfer AS rlautotransfer,
                       payloadb1personentityid AS b1personentityid,
                       payloadb1ficoscore AS b1ficoscore,
                       payloadb2personentityid AS b2personentityid,
                       payloadb2ficoscore AS b2ficoscore,
                       payloadhtisprimaryresidence AS isprimaryresidence,
                       payloadhtgoals AS leadgoals, 
                       payloadpropertyuse AS propertyuse,
                       payloadhtlbtimeframetobuy AS lbtimeframetobuy,
                       payloadhtwebcredit AS selfassessedcredit,
                       payloadhasrealestateagent AS hasrealestateagent,
                       payloadhtfname AS fname,
                       payloadhtmname AS mname,
                       payloadhtlname AS lname,
                       payloadhtemail AS email,
                       payloadhtgcid AS gcid,
                       payloadhtgcid2 AS gcid2,
                       datekey
                  FROM :table_name;
                  WHERE datekey >= :start_date;
                  AND datekey <= :end_date;)
                  SELECT *
                  FROM T
                  WHERE 1 = 1
                  """
        if self.lead_type:
            query += """ AND LOWER(typeoflead) = '{}' """.format(self.lead_type)
        if self.loan_purpose:
            query += """ AND loanpurpose = '{}' """.format(self.loan_purpose)
        query += """ LIMIT :limit; """            
        return query

    
class QueryAdobeEvents(QueryDataBase):
    
    """A child class which inherits all methods and properties from QueryDataBase used to represent adobe events query
    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'adobe_websessionpostevent'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    event_codes : list
        Filter data with a list of adobe event code
    first_only : boolean
        If True, return data with the first occured event time for each event in each unique visit. Default to False
    """
    
    def __init__(self, start_date, end_date, table_name='adobe_websessionpostevent', limit='ALL', event_codes=[], first_only=False, **kwargs):
        
        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
        self.event_codes = event_codes
        self.first_only = first_only
        
    def sql_instruction(self):
        
        """Queries data based on provided parameters.
        Parameters
        ----------
        self : object
            QueryAdobeEvents class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: date_time(EST), uniquevisitkey, mcvisid_visitorid, eventid
        """
        
        query = """WITH T AS
                   (
                   SELECT DISTINCT 
                       date_time,
                       uniquevisitkey,
                       mcvisid_visitorid,
                       eventid,
                       ROW_NUMBER() OVER 
                       (
                       PARTITION BY 
                           uniquevisitkey,
                           mcvisid_visitorid,
                           eventid
                       ORDER BY 
                           date_time
                       ) AS row_number
                   FROM :table_name;
                   WHERE datekey >= :start_date;
                   AND datekey <= :end_date;
                   """
        if len(self.event_codes) > 0:
            query += """ AND eventid IN ('{}') """.format("', '".join(self.event_codes))
        query +=""") 
                   SELECT date_time,
                          uniquevisitkey,
                          mcvisid_visitorid,
                          eventid
                   FROM T
                   WHERE 1 = 1
                   """
        if self.first_only:
            query += """ AND row_number = 1 """
        query += """ LIMIT :limit; """
        
        return query
    
    
class QueryAdobeVisitId(QueryDataBase):
    
    """A child class which inherits all methods and properties from QueryDataBase used to represent adobe visitor id query

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'adobe_dq_prcd_data'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    """
    
    def __init__(self, start_date, end_date, table_name='adobe_dq_prcd_data', limit='ALL', **kwargs):
        
        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """  
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
       
    def sql_instruction(self):
        
        """Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryAdobeVisitId class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: min_visit_date_time(EST), uniquevisitkey, mcvisid_visitorid, 
            rocketaccountid_evar5, rocketmortgage_loannumber_evar8, rocketmortgage_loanguid_evar115, datekey
        """
        
        query = """WITH Adobe AS
                   (
                   SELECT DISTINCT 
                       date_time,
                       uniquevisitkey,
                       mcvisid_visitorid,
                       rocketaccountid_evar5,
                       rocketmortgage_loannumber_evar8,
                       rocketmortgage_loanguid_evar115,
                       datekey
                   FROM :table_name;
                   WHERE datekey >= :start_date;
                   AND datekey <= :end_date;
                   )
                   SELECT MIN(date_time) AS min_visit_date_time,
                          uniquevisitkey,
                          mcvisid_visitorid,
                          MAX(rocketaccountid_evar5) AS rocketaccountid_evar5,
                          MAX(rocketmortgage_loannumber_evar8) AS rocketmortgage_loannumber_evar8,
                          MAX(rocketmortgage_loanguid_evar115) AS rocketmortgage_loanguid_evar115,
                          MIN(datekey) AS datekey
                   FROM Adobe
                   GROUP BY 
                       uniquevisitkey,
                       mcvisid_visitorid
                   LIMIT :limit;
                   """
        return query
    
    
class QueryAdobeVisitInfo(QueryDataBase):
    
    """A child class which inherits all methods and properties from QueryDataBase used to represent adobe visit information query

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'adobe_dq_prcd_data'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    site_sections : list
        Filter data with selected site sections such as 'ql lander', 'rocket lander' etc
    loan_purpose : str
        Filter data with loan purpose 'Refinance' or 'Purchase'
    group_id : str
        ID for experiment group, for instance '562585:0:0'   
    """
    
    def __init__(self, start_date, end_date, table_name='adobe_dq_prcd_data', limit='ALL', site_sections=[], loan_purpose='', group_id='', **kwargs):
        
        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """  
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
        self.site_sections = site_sections
        self.loan_purpose = loan_purpose
        self.group_id = group_id
        self.condition_map = self._filter_condition_map()
        
    def _filter_condition_map(self):
        
        """Create a dictionary that maps input filter conditions with WHERE clause conditions in sql query

        Parameters
        ----------
        self : object
            QueryAdobeLeadformInput class instance
            
        Returns
        -------
        condition_map : dict
            Return a dictionary of filter conditions with condition names
        """
        
        condition_map = {}
        if self.loan_purpose:
            condition_map['loanpurpose_condition'] = """ AND (
                                                         LOWER(post_mvvar3) LIKE '%--purpose:{}%' OR
                                                         LOWER(post_mvvar3) LIKE 'purpose:{}%' OR 
                                                         evar40 = '{}'
                                                         ) """.format(self.loan_purpose.lower(), self.loan_purpose.lower(), self.loan_purpose)
        if self.site_sections:
            condition_map['sitesection_condition'] = """ AND sitesection IN ('{}') """.format("', '".join(self.site_sections))
        if self.group_id:
            condition_map['groupid_condition'] = """ AND (
                                                     post_tnt = '{}' OR
                                                     post_tnt LIKE '%--{}%' OR
                                                     post_tnt LIKE '{}%'
                                                     ) """.format(self.group_id, self.group_id, self.group_id)
        return condition_map
       
    def sql_instruction(self):
        
        """Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryAdobeVisitInfo class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: uniquevisitkey, mcvisid_visitorid, visitnumber, 
            min/max_visit_date_time(EST), first_sitesection, first_loanpurpose, first_pagename, first_qls, first_va_finder_id, 
            first_va_closer_id, first_referrer, first_geo_city, first_geo_zip, first_carrier, first_domain
        """
        
        if self.condition_map:
            for idx, key in enumerate(self.condition_map.keys()):
                if idx == 0:
                    query = """ WITH T0 AS
                                (
                                SELECT DISTINCT
                                    uniquevisitkey,
                                    mcvisid_visitorid
                                FROM :table_name;
                                WHERE datekey >= :start_date;
                                AND datekey <= :end_date;
                                """
                    query += self.condition_map[key]
                    sub_query =""" ), Adobe_id AS
                                   (
                                   SELECT DISTINCT
                                       T0.uniquevisitkey,
                                       T0.mcvisid_visitorid
                                   FROM T0
                                   """
                else:
                    query +=""" ), T{} AS
                                (
                                SELECT DISTINCT
                                    uniquevisitkey,
                                    mcvisid_visitorid
                                FROM :table_name;
                                WHERE datekey >= :start_date;
                                AND datekey <= :end_date;
                                """.format(idx)
                    query += self.condition_map[key]
                    sub_query += """ JOIN T{} ON
                                         T0.uniquevisitkey = T{}.uniquevisitkey
                                         AND T0.mcvisid_visitorid = T{}.mcvisid_visitorid
                                         """.format(idx, idx, idx)
            query += sub_query
        else:
            query = """ WITH Adobe_id AS 
                        (
                        SELECT DISTINCT
                            uniquevisitkey,
                            mcvisid_visitorid
                        FROM :table_name;
                        WHERE datekey >= :start_date;
                        AND datekey <= :end_date;
                        """
        query+= """ ),
                    Adobe AS 
                    (
                    SELECT DISTINCT 
                        A.date_time,
                        A.uniquevisitkey,
                        A.mcvisid_visitorid,
                        A.visitnumber,
                        A.rocketaccountid_evar5,
                        A.rocketmortgage_loannumber_evar8,
                        A.rocketmortgage_loanguid_evar115,
                        LAST_VALUE(NULLIF(sitesection, '')) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            sitesection
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_sitesection,
                        CASE WHEN sitesection = 'ql lander' 
                             THEN 1 
                             ELSE 0 
                        END AS visit_ql_lander,
                        CASE WHEN sitesection = 'rocket lander' 
                             THEN 1 
                             ELSE 0 
                        END AS visit_rocket_lander,
                        CASE WHEN 
                             (
                             LOWER(post_mvvar3) LIKE '%--purpose:purchase%' OR
                             LOWER(post_mvvar3) LIKE 'purpose:purchase%' OR 
                             evar40 = 'Purchase'
                             ) 
                             THEN 'Purchase'
                             WHEN 
                             (
                             LOWER(post_mvvar3) LIKE '%--purpose:refinance%' OR 
                             LOWER(post_mvvar3) LIKE 'purpose:refinance%' OR 
                             evar40 = 'Refinance'
                             ) 
                             THEN 'Refinance'
                             WHEN evar40 != '' 
                             THEN evar40
                        END AS loanpurpose,
                        LAST_VALUE(NULLIF(pagename, '')) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            pagename 
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_pagename,
                        LAST_VALUE(NULLIF(qls, '')) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            qls
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_qls,
                        LAST_VALUE(va_finder_id) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            va_finder_id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_va_finder_id,
                        LAST_VALUE(va_closer_id) IGNORE NULLS OVER
                        (PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            va_closer_id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_va_closer_id,
                        LAST_VALUE(NULLIF(post_referrer, '')) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid
                        ORDER BY 
                            date_time DESC,
                            post_referrer
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_referrer,
                        LAST_VALUE(NULLIF(country, '')) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            country
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_geo_country,
                        LAST_VALUE(NULLIF(region, '')) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            region
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_geo_region,
                        LAST_VALUE(NULLIF(city, '')) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            city
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_geo_city,
                        LAST_VALUE(zip) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            zip
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_geo_zip,
                        LAST_VALUE(NULLIF(carrier, '')) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid
                        ORDER BY 
                            date_time DESC,
                            carrier
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_carrier,
                        LAST_VALUE(NULLIF(domain, '')) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            A.uniquevisitkey, 
                            A.mcvisid_visitorid
                        ORDER BY 
                            date_time DESC,
                            domain
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_domain
                    FROM :table_name; A
                    JOIN Adobe_id I ON 
                        A.uniquevisitkey = I.uniquevisitkey AND
                        A.mcvisid_visitorid = I.mcvisid_visitorid
                    WHERE datekey >= :start_date;
                    AND datekey <= :end_date;
                    ),
                    A AS 
                    (
                    SELECT *,
                        LAST_VALUE(loanpurpose) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            loanpurpose
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_loanpurpose,
                        LAST_VALUE(loanpurpose) IGNORE NULLS OVER
                        (
                        PARTITION BY
                            uniquevisitkey, 
                            mcvisid_visitorid 
                        ORDER BY 
                            date_time,
                            loanpurpose
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_loanpurpose
                    FROM Adobe
                    )
                    SELECT uniquevisitkey,
                           mcvisid_visitorid,
                           visitnumber,
                           MIN(date_time) AS min_visit_date_time,
                           MAX(date_time) AS max_visit_date_time,
                           MAX(rocketaccountid_evar5) AS rocketaccountid_evar5,
                           MAX(rocketmortgage_loannumber_evar8) AS rocketmortgage_loannumber_evar8,
                           MAX(rocketmortgage_loanguid_evar115) AS rocketmortgage_loanguid_evar115,
                           MAX(first_sitesection) AS first_sitesection,
                           MAX(visit_ql_lander) AS visit_ql_lander,
                           MAX(visit_rocket_lander) AS visit_rocket_lander,
                           MAX(first_loanpurpose) AS first_loanpurpose,
                           MAX(last_loanpurpose) AS last_loanpurpose,
                           MAX(first_pagename) AS first_pagename,
                           MAX(first_qls) AS first_qls,
                           MAX(first_va_finder_id) AS first_va_finder_id,
                           MAX(first_va_closer_id) AS first_va_closer_id,
                           MAX(first_referrer) AS first_referrer,
                           MAX(first_geo_country) AS first_geo_country,
                           MAX(first_geo_region) AS first_geo_region,
                           MAX(first_geo_city) AS first_geo_city,
                           MAX(first_geo_zip) AS first_geo_zip,
                           MAX(first_carrier) AS first_carrier,
                           MAX(first_domain) AS first_domain
                    FROM A
                    GROUP BY 
                        uniquevisitkey, 
                        mcvisid_visitorid, 
                        visitnumber
                    LIMIT :limit;
                    """
        return query
    
    
class QueryAdobeLeadformInput(QueryDataBase):
    
    """A child class which inherits all methods and properties from QueryDataBase used to represent adobe lead form input query

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'adobe_dq_prcd_data'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    site_sections : list
        Filter data with selected site sections. By default, this is set to ['ql lander', 'rocket lander']
    loan_purpose : str
        Filter data with loan purpose 'Refinance' or 'Purchase'
    group_id : str
        ID for experiment group, for instance '562585:0:0'
    """
    
    def __init__(self, start_date, end_date, table_name='adobe_dq_prcd_data', limit='ALL', site_sections=['ql lander', 'rocket lander'], loan_purpose='', group_id='', **kwargs):
        
        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """  
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
        self.site_sections = site_sections
        self.loan_purpose = loan_purpose
        self.group_id = group_id
        self.condition_map = self._filter_condition_map()
        
    def _filter_condition_map(self):
        
        """Create a dictionary that maps input filter conditions with WHERE clause conditions in sql query

        Parameters
        ----------
        self : object
            QueryAdobeLeadformInput class instance
            
        Returns
        -------
        condition_map : dict
            Return a dictionary of filter conditions with condition names
        """
        
        condition_map = {}
        if self.loan_purpose:
            condition_map['loanpurpose_condition'] = """ AND (
                                                         LOWER(post_mvvar3) LIKE '%--purpose:{}%' OR
                                                         LOWER(post_mvvar3) LIKE 'purpose:{}%' OR 
                                                         evar40 = '{}'
                                                         ) """.format(self.loan_purpose.lower(), self.loan_purpose.lower(), self.loan_purpose)
        if self.site_sections:
            condition_map['sitesection_condition'] = """ AND sitesection IN ('{}') """.format("', '".join(self.site_sections))
        if self.group_id:
            condition_map['groupid_condition'] = """ AND (
                                                     post_tnt = '{}' OR
                                                     post_tnt LIKE '%--{}%' OR
                                                     post_tnt LIKE '{}%'
                                                     ) """.format(self.group_id, self.group_id, self.group_id)
        return condition_map
       
    def sql_instruction(self):
        
        """Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryAdobeLeadformInput class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: uniquevisitkey, mcvisid_visitorid, visitnumber, 
            min/max_visit_date_time(EST), first_landing_sitesection, min_landingpage_date_time(EST), first/last_homedescription, 
            first/last_creditrating, first/last_timeframetopurchase, first/last_firsttimebuyer, first/last_hasrealestateagent, 
            first/last_purchaseprice
        """
        
        if self.condition_map:
            for idx, key in enumerate(self.condition_map.keys()):
                if idx == 0:
                    query =     """ 
                                WITH T0 AS
                                (
                                SELECT DISTINCT
                                    uniquevisitkey,
                                    mcvisid_visitorid
                                FROM :table_name;
                                WHERE datekey >= :start_date;
                                AND datekey <= :end_date;
                                """
                    
                    query += self.condition_map[key]
                    
                    sub_query = """
                                ), Adobe_id AS
                                   (
                                   SELECT DISTINCT
                                       T0.uniquevisitkey,
                                       T0.mcvisid_visitorid
                                   FROM T0
                                """
                else:
                    query +=    """ 
                                ), T{} AS
                                (
                                SELECT DISTINCT
                                    uniquevisitkey,
                                    mcvisid_visitorid
                                FROM :table_name;
                                WHERE datekey >= :start_date;
                                AND datekey <= :end_date;
                                """.format(idx)
                    
                    query += self.condition_map[key]
                    
                    sub_query += """ 
                                JOIN T{} ON
                                    T0.uniquevisitkey = T{}.uniquevisitkey
                                    AND T0.mcvisid_visitorid = T{}.mcvisid_visitorid """.format(idx, idx, idx)
            query += sub_query
        else:
            query = """ 
                    WITH Adobe_id AS 
                        (
                        SELECT DISTINCT
                            uniquevisitkey,
                            mcvisid_visitorid
                        FROM :table_name;
                        WHERE datekey >= :start_date;
                        AND datekey <= :end_date;
                    """
        query +=    """ ),
                    Adobe AS 
                    (
                    SELECT DISTINCT
                        A.uniquevisitkey,
                        A.mcvisid_visitorid,
                        A.visitnumber,
                        A.date_time,
                        A.rocketaccountid_evar5,
                        A.rocketmortgage_loannumber_evar8,
                        A.rocketmortgage_loanguid_evar115,
                        CASE WHEN A.rocketaccountid_evar5 IS NOT NULL AND
                                  A.rocketaccountid_evar5 <> '' 
                             THEN A.date_time 
                        END AS rocketaccountid_date_time,
                        CASE WHEN A.rocketmortgage_loannumber_evar8 IS NOT NULL AND
                                  A.rocketmortgage_loannumber_evar8 <> '' 
                             THEN A.date_time 
                        END AS loannumber_date_time,
                        CASE WHEN A.rocketmortgage_loanguid_evar115 IS NOT NULL AND
                                  A.rocketmortgage_loanguid_evar115 <> '' 
                             THEN A.date_time 
                        END AS loanguid_date_time,
                        CASE WHEN 
                             (
                             LOWER(post_mvvar3) LIKE '%--purpose:purchase%' OR
                             LOWER(post_mvvar3) LIKE 'purpose:purchase%' OR 
                             evar40 = 'Purchase'
                             )
                             THEN 'Purchase'
                             WHEN 
                             (
                             LOWER(post_mvvar3) LIKE '%--purpose:refinance%' OR 
                             LOWER(post_mvvar3) LIKE 'purpose:refinance%' OR 
                             evar40 = 'Refinance'
                             ) 
                             THEN 'Refinance'
                             WHEN evar40 != '' 
                             THEN evar40 
                        END AS loanpurpose,
                        CASE WHEN sitesection = 'ql lander' OR 
                                  sitesection = 'rocket lander' 
                             THEN sitesection 
                        END AS landing_sitesection,
                        CASE WHEN sitesection = 'ql lander' OR
                                  sitesection = 'rocket lander' 
                             THEN A.date_time
                        END AS landingpage_date_time,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'homedescription:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'homedescription:(.*)$', 1)
                                  ) 
                        END AS homedescription,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'propertyuse:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'propertyuse:(.*)$', 1)
                                  ) 
                        END AS propertyuse,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'creditrating:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'creditrating:(.*)$', 1)
                                  ) 
                        END AS creditrating,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(REGEXP_REPLACE(LOWER(post_mvvar3), '--'), '(loanpurposepurchase|timeframetopurchase):([^\*]+)', 2), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), '(loanpurposepurchase|timeframetopurchase):(.*)$', 2)
                                  ) 
                        END AS timeframetopurchase,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'firsttimebuyer:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'firsttimebuyer:(.*)$', 1)
                                  ) 
                        END AS firsttimebuyer,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'hasrealestateagent:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'hasrealestateagent:(.*)$', 1)
                                  )
                        END AS hasrealestateagent,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'purchaseprice:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'purchaseprice:(.*)$', 1)
                                  )
                        END AS purchaseprice,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'downpayment:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'downpayment:(.*)$', 1)
                                  )
                        END AS downpayment,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'homevalue:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'homevalue:(.*)$', 1)
                                  )
                        END AS homevalue,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'mortgagebalance:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'mortgagebalance:(.*)$', 1)
                                  )
                        END AS mortgagebalance,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'loanpurposerefinancecashout:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'loanpurposerefinancecashout:(.*)$', 1)
                                  )
                        END AS refinancereason,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'secondmortgage:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'secondmortgage:(.*)$', 1)
                                  )
                        END AS secondmortgage,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'employmentstatus:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'employmentstatus:(.*)$', 1)
                                  )
                        END AS employmentstatus,
                        CASE WHEN sitesection IN ('ql lander', 'rocket lander') 
                             THEN COALESCE
                                  (
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'bankruptcy:([^-\*]+)', 1), 
                                  REGEXP_EXTRACT(LOWER(post_mvvar3), 'bankruptcy:(.*)$', 1)
                                  )                   
                        END AS bankruptcy,
                        REGEXP_EXTRACT(post_tnt, '(\d+:[^0]:0)') AS first_testexperiment,
                        post_tnt,
                        CAST(va_finder_id AS VARCHAR) AS va_finder_id,
                        CAST(va_closer_id AS VARCHAR) AS va_closer_id
                    FROM :table_name; A
                    JOIN Adobe_id I ON 
                        A.uniquevisitkey = I.uniquevisitkey AND
                        A.mcvisid_visitorid = I.mcvisid_visitorid
                    WHERE A.datekey >= :start_date; and A.datekey <= :end_date;
                    """
        
        if self.condition_map:
            for idx, key in enumerate(self.condition_map.keys()):
                query += self.condition_map[key]

        query +=    """
                    ),
                    A AS 
                    (
                    SELECT *,
                        LAST_VALUE(landing_sitesection) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            landing_sitesection
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_landing_sitesection,
                        LAST_VALUE(loanpurpose) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            loanpurpose
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_loanpurpose,
                        LAST_VALUE(loanpurpose) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            loanpurpose
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_loanpurpose,
                        LAST_VALUE(homedescription) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            homedescription
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_homedescription,
                        LAST_VALUE(homedescription) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            homedescription
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_homedescription,
                        LAST_VALUE(propertyuse) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            propertyuse
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_propertyuse,
                        LAST_VALUE(propertyuse) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            propertyuse
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_propertyuse,
                        LAST_VALUE(creditrating) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            creditrating
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_creditrating,
                        LAST_VALUE(creditrating) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            creditrating
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_creditrating,
                        LAST_VALUE(timeframetopurchase) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            timeframetopurchase
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_timeframetopurchase,
                        LAST_VALUE(timeframetopurchase) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            timeframetopurchase
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_timeframetopurchase,
                        LAST_VALUE(firsttimebuyer) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            firsttimebuyer
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_firsttimebuyer,
                        LAST_VALUE(firsttimebuyer) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            firsttimebuyer
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_firsttimebuyer,
                        LAST_VALUE(hasrealestateagent) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            hasrealestateagent
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_hasrealestateagent,
                        LAST_VALUE(hasrealestateagent) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            hasrealestateagent
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_hasrealestateagent,
                        LAST_VALUE(purchaseprice) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            purchaseprice
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_purchaseprice,
                        LAST_VALUE(purchaseprice) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            purchaseprice
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_purchaseprice,
                        LAST_VALUE(downpayment) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            downpayment
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_downpayment,
                        LAST_VALUE(downpayment) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            downpayment
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_downpayment,
                        LAST_VALUE(homevalue) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            homevalue
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_homevalue,
                        LAST_VALUE(homevalue) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            homevalue
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_homevalue,
                        LAST_VALUE(mortgagebalance) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            mortgagebalance
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_mortgagebalance,
                        LAST_VALUE(mortgagebalance) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            mortgagebalance
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_mortgagebalance,
                        LAST_VALUE(refinancereason) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            refinancereason
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_refinancereason,
                        LAST_VALUE(refinancereason) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            refinancereason
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_refinancereason,
                        LAST_VALUE(secondmortgage) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            secondmortgage
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_secondmortgage,
                        LAST_VALUE(secondmortgage) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            secondmortgage
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_secondmortgage,
                        LAST_VALUE(employmentstatus) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            employmentstatus
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_employmentstatus,
                        LAST_VALUE(employmentstatus) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            employmentstatus
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_employmentstatus,
                        LAST_VALUE(bankruptcy) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time DESC,
                            bankruptcy
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_bankruptcy,
                        LAST_VALUE(bankruptcy) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid  
                        ORDER BY 
                            date_time,
                            bankruptcy
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS last_bankruptcy,
                        CASE WHEN first_testexperiment IS NOT NULL 
                             THEN date_time 
                        END AS testexperiment_date_time,
                        LAST_VALUE(va_finder_id) IGNORE NULLS OVER
                        (
                        PARTITION BY 
                            uniquevisitkey,
                            mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            va_finder_id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_va_finder_id,
                        LAST_VALUE(va_closer_id) IGNORE NULLS OVER
                        (PARTITION BY 
                            uniquevisitkey, 
                            mcvisid_visitorid 
                        ORDER BY 
                            date_time DESC,
                            va_closer_id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS first_va_closer_id
                        FROM Adobe
                        )
                    SELECT uniquevisitkey,
                           mcvisid_visitorid,
                           visitnumber,
                           MIN(date_time) AS min_visit_date_time,
                           MAX(date_time) AS max_visit_date_time,
                           MAX(rocketaccountid_evar5) AS rocketaccountid_evar5,
                           MAX(rocketmortgage_loannumber_evar8) AS rocketmortgage_loannumber_evar8,
                           MAX(rocketmortgage_loanguid_evar115) AS rocketmortgage_loanguid_evar115,
                           MIN(rocketaccountid_date_time) AS min_rocketaccountid_date_time,
                           MIN(loannumber_date_time) AS min_loannumber_date_time,
                           MIN(loanguid_date_time) AS min_loanguid_date_time,
                           MAX(first_loanpurpose) AS first_loanpurpose,
                           MAX(last_loanpurpose) AS last_loanpurpose,
                           MAX(first_landing_sitesection) AS first_landing_sitesection,
                           MIN(landingpage_date_time) AS min_landingpage_date_time,
                           MAX(landingpage_date_time) AS max_landingpage_date_time,
                           MAX(first_homedescription) AS first_homedescription,
                           MAX(last_homedescription) AS last_homedescription,
                           MAX(first_propertyuse) AS first_propertyuse,
                           MAX(last_propertyuse) AS last_propertyuse,
                           MAX(first_creditrating) AS first_creditrating,
                           MAX(last_creditrating) AS last_creditrating,
                           MAX(first_timeframetopurchase) AS first_timeframetopurchase,
                           MAX(last_timeframetopurchase) AS last_timeframetopurchase,
                           MAX(first_firsttimebuyer) AS first_firsttimebuyer,
                           MAX(last_firsttimebuyer) AS last_firsttimebuyer,
                           MAX(first_hasrealestateagent) AS first_hasrealestateagent,
                           MAX(last_hasrealestateagent) AS last_hasrealestateagent,
                           MAX(first_purchaseprice) AS first_purchaseprice,
                           MAX(last_purchaseprice) AS last_purchaseprice,
                           MAX(first_downpayment) AS first_downpayment,
                           MAX(last_downpayment) AS last_downpayment,                           
                           MAX(first_homevalue) AS first_homevalue,
                           MAX(last_homevalue) AS last_homevalue,
                           MAX(first_mortgagebalance) AS first_mortgagebalance,
                           MAX(last_mortgagebalance) AS last_mortgagebalance,
                           MAX(first_refinancereason) AS first_refinancereason,
                           MAX(last_refinancereason) AS last_refinancereason,
                           MAX(first_secondmortgage) AS first_secondmortgage,
                           MAX(last_secondmortgage) AS last_secondmortgage,
                           MAX(first_employmentstatus) AS first_employmentstatus,
                           MAX(last_employmentstatus) AS last_employmentstatus,
                           MAX(first_bankruptcy) AS first_bankruptcy,
                           MAX(last_bankruptcy) AS last_bankruptcy,
                           MAX(first_testexperiment) AS first_testexperiment,
                           MAX(post_tnt) AS testexperiment,
                           MIN(testexperiment_date_time) AS min_testexperiment_date_time,
                           MAX(testexperiment_date_time) AS max_testexperiment_date_time,
                           MAX(first_va_finder_id) AS first_va_finder_id,
                           MAX(first_va_closer_id) AS first_va_closer_id
                    FROM A
                    GROUP BY 
                        uniquevisitkey,
                        mcvisid_visitorid,
                        visitnumber
                    LIMIT :limit;
                    """
        return query
    
    
class QueryLoanIdentifier(QueryDataBase):
    
    """A child class which inherits all methods and properties from QueryDataBase used to represent loan identifier query

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'loan_identifier_dim'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    """
    
    def __init__(self, start_date, end_date, table_name='loan_identifier_dim', limit='ALL', **kwargs):
        
        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """   
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
       
    def sql_instruction(self):
        
        """Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryLoanIdentifier class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: loanidentifierdimsk, loannumber, loanguid
        """
        
        query = """WITH T AS
                   (
                   SELECT DISTINCT
                       loanidentifierdimsk,
                       loannumber,
                       LOWER(loanguid) as loanguid
                   FROM :table_name;
                   WHERE sourcecreatedatetime 
                       BETWEEN timestamp :start_date;
                       AND (timestamp :end_date; + interval '1' day)
                       --AND iscurrentrecordind = true
                   ),
                   T1 AS
                   (
                   SELECT DISTINCT
                       loanidentifierdimsk
                   FROM T
                   ),
                   T2 AS
                   (
                   SELECT DISTINCT
                       loanidentifierdimsk,
                       loannumber
                   FROM T
                   WHERE loannumber IS NOT NULL
                   AND loannumber <> ''
                   ),
                   T3 AS
                   (
                   SELECT DISTINCT
                       loanidentifierdimsk,
                       loanguid
                   FROM T
                   WHERE loanguid IS NOT NULL
                   AND loanguid <> ''
                   )
                   SELECT T1.loanidentifierdimsk,
                          T2.loannumber,
                          T3.loanguid
                   FROM T1
                   LEFT JOIN T2 ON
                       T1.loanidentifierdimsk = T2.loanidentifierdimsk
                   LEFT JOIN T3 ON
                       T1.loanidentifierdimsk = T3.loanidentifierdimsk
                   LIMIT :limit; 
                   """
        return query
    
    
class QueryMajorMilestone(QueryDataBase):
    
    """A child class which inherits all methods and properties from QueryDataBase used to represent major milestone query

    Attributes
    ----------
    start_date : str
        Start date for the experiment in 'yyyy-mm-dd' format 
    end_date : str
        End date for the experiment in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving experiment data. By default, this is set to 'loan_major_milestone_fact'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    milestones : list
        Filter data with selected milestones. By default, this is set to ['Lead','Net Leads', 'Allocated', 
            'Credit', 'PAL', 'VAL', 'Application', 'Folder', 'Closing']
    first_only : boolean
        If True, return data with the first occured milestone time for each milestone in each unique loanidentifier. Default to True   
    """
    
    def __init__(self, start_date, end_date, table_name='loan_major_milestone_fact', limit='ALL', milestones=['Lead',
                                                                                                              'Net Leads',
                                                                                                              'Allocated',
                                                                                                              'Credit',
                                                                                                              'PAL',
                                                                                                              'VAL',
                                                                                                              'Application',
                                                                                                              'Folder',
                                                                                                              'Closing'
                                                                                                             ], first_only=True, **kwargs):
        
        """super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """   
        super().__init__(start_date, end_date, table_name, limit, **kwargs)
        self.milestone_sk_list = self._get_milestone_sk(milestones)
        self.first_only = first_only
        
    def _get_milestone_sk(self, milestones):
        
        """Map milestones names to sk ids

        Parameters
        ----------
        self : object
            QueryMajorMilestone class instance
        milestones : list
            Input milestones list
        
        Returns
        -------
        list : list
            List of input milestone sk ids
        """
        
        if 'Lead' not in milestones:
            milestones = milestones + ['Lead']
            
        milestone_sk_mapping = {'7' : 'Lead',
                                '6' : 'Lead',
                                '14': 'Net Leads',
                                '17': 'Net Leads',
                                '9': 'Allocated',
                                '121': 'Allocated',
                                '1': 'Credit',
                                '2': 'Credit',
                                '124': 'Credit',
                                '120': 'Credit',
                                '131': 'Credit',
                                '132': 'Credit',
                                '133': 'Credit',
                                '3': 'PAL',
                                '129': 'VAL',
                                '13': 'Rate Look',
                                '80': 'Application',
                                '5': 'Setup',
                                '4': 'Folder',
                                '11': 'Conditionally Approved',
                                '12': 'Final Signoff',
                                '10': 'Scheduled Closing',
                                '77': 'HUD Review Complete',
                                '8': 'Closing',
                                '74': 'Loans Disburserd',
                                '75': 'Loans Funded',
                                '14': 'Suspense'}
        return [k for k, v in milestone_sk_mapping.items() if v in milestones]
       
    def sql_instruction(self):
        
        """Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryMajorMilestone class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: loanidentifierdimsk, loanmilstonesk, 
            loanmilestone_eventname, loanmilestone_groupname, datetime(EST)
        """
        
        query ="""WITH T AS
                  (
                  SELECT DISTINCT
                      loanidentifierdimsk,
                      loanmilestonesk,
                      CASE WHEN loanmilestonesk = 7 
                           THEN 'LOLA Lead Created'
                           WHEN loanmilestonesk = 6 
                           THEN 'Rocket Lead Created'
                           WHEN loanmilestonesk = 14 
                           THEN 'Net Leads'
                           WHEN loanmilestonesk = 17 
                           THEN 'Rocket Net Leads'
                           WHEN loanmilestonesk = 9 
                           THEN 'Allocated'
                           WHEN loanmilestonesk = 121 
                           THEN 'Salesforce Allocated'
                           WHEN loanmilestonesk = 1 
                           THEN 'Idle Pulled Credit'
                           WHEN loanmilestonesk = 2 
                           THEN 'Rocket Credit Pulled'
                           WHEN loanmilestonesk = 124 
                           THEN 'Credit Pulled in AMP'
                           WHEN loanmilestonesk = 120
                           THEN 'Rocket Logic Credit Pulled'
                           WHEN loanmilestonesk = 131 
                           THEN 'Soft Credit Pulled'
                           WHEN loanmilestonesk = 132 
                           THEN 'Rocket Logic Soft Credit Pulled'
                           WHEN loanmilestonesk = 133
                           THEN 'RMA Soft Credit Pulled'
                           WHEN loanmilestonesk = 3
                           THEN 'PAL'
                           WHEN loanmilestonesk = 129 
                           THEN 'VAL'
                           WHEN loanmilestonesk = 13
                           THEN 'Rate Lock'
                           WHEN loanmilestonesk = 80
                           THEN 'Application'
                           WHEN loanmilestonesk = 5
                           THEN 'LKWD Setup'
                           WHEN loanmilestonesk = 4
                           THEN 'LKWD Folder'
                           WHEN loanmilestonesk = 11
                           THEN 'Conditionally Approved'
                           WHEN loanmilestonesk = 12
                           THEN 'Final Signoff'
                           WHEN loanmilestonesk = 10
                           THEN 'Scheduled Closing'
                           WHEN loanmilestonesk = 77
                           THEN 'HUD Review Complete'
                           WHEN loanmilestonesk = 8
                           THEN 'LKWD Closing'
                           WHEN loanmilestonesk = 74
                           THEN 'Draft Honored - Warehoused'
                           WHEN loanmilestonesk = 15
                           THEN 'Funded by Investor'
                           WHEN loanmilestonesk = 16
                           THEN 'Suspense'  
                      END AS loanmilestone_eventname,
                      CASE WHEN loanmilestonesk IN (7, 6) 
                           THEN 'Lead'
                           WHEN loanmilestonesk IN (14, 17) 
                           THEN 'Net Leads'
                           WHEN loanmilestonesk IN (9, 121)
                           THEN 'Allocated'
                           WHEN loanmilestonesk IN (1, 124, 120)
                           THEN 'Offline Credit'
                           WHEN loanmilestonesk = 2
                           THEN 'Online Credit'
                           WHEN loanmilestonesk IN (131, 132, 133)
                           THEN 'Soft Credit'
                           WHEN loanmilestonesk = 3 
                           THEN 'PAL'
                           WHEN loanmilestonesk = 129 
                           THEN 'VAL'
                           WHEN loanmilestonesk = 13 
                           THEN 'Rate Look'
                           WHEN loanmilestonesk = 80 
                           THEN 'Application'
                           WHEN loanmilestonesk = 5 
                           THEN 'Setup'
                           WHEN loanmilestonesk = 4 
                           THEN 'Folder'
                           WHEN loanmilestonesk = 11 
                           THEN 'Conditionally Approved'
                           WHEN loanmilestonesk = 12 
                           THEN 'Final Signoff'
                           WHEN loanmilestonesk = 10 
                           THEN 'Scheduled Closing'
                           WHEN loanmilestonesk = 77 
                           THEN 'HUD Review Complete'
                           WHEN loanmilestonesk = 8 
                           THEN 'Closing'
                           WHEN loanmilestonesk = 74 
                           THEN 'Loans Disburserd'
                           WHEN loanmilestonesk = 75
                           THEN 'Loans Funded'
                           WHEN loanmilestonesk = 16 
                           THEN 'Suspense'
                      END AS loanmilestone_groupname,
                      eventdatetime AS estdatetime,
                      ROW_NUMBER() OVER 
                      (
                      PARTITION BY 
                          loanidentifierdimsk,
                          loanmilestonesk
                      ORDER BY 
                          eventdatetime
                      ) AS row_number
                  FROM :table_name;
                  WHERE eventdatetime 
                      BETWEEN timestamp :start_date;
                      AND (timestamp :end_date; + interval '1' day)
                  AND iscurrentrecordind = true
                  """
        if self.milestone_sk_list:
            query += """ AND loanmilestonesk IN ({}) """.format(', '.join(self.milestone_sk_list))
            
        query+=""")
                  SELECT loanidentifierdimsk,
                         loanmilestonesk,
                         loanmilestone_eventname,
                         loanmilestone_groupname,
                         estdatetime
                  FROM T
                  WHERE 1 = 1
                  """
        if self.first_only:
            query += """ AND row_number = 1 """
            
        query += """ LIMIT :limit; """
        
        return query

class QueryCallCommunication(QueryDataBase):
    
    """
    A child class which inherits all methods and properties from QueryDataBase used to represent communication data query

    Attributes
    ----------
    start_date : str
        Start date for the communication in 'yyyy-mm-dd' format 
    end_date : str
        End date for the communication in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving communication data. By default, this is set to 'communication_call'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    """
    
    def __init__(self, start_date, end_date, table_name='communication_call', limit='ALL', direction=['outgoing', 'incoming'], **kwargs):
        
        """
        super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """   
        super().__init__(start_date, end_date, table_name='communication_call', limit='ALL', **kwargs)
        
        self.direction = direction
        if isinstance(self.direction, list) is False:
            self.direction = list(direction.split(','))
       
    def sql_instruction(self):
        
        """
        Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryCallCommunication class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: 
                communication_direction, rktpersonid, loan_number, communication_direction, communication_from_phone_number
        """
        
        query = """
                WITH call_communication AS
                    (
                    SELECT
                        CASE
                            WHEN trim(lower(communication_direction))='outgoing'
                            THEN communication_to_party_guid
                            ELSE communication_from_party_guid
                            END AS rktpersonid,
                        communication_guid,
                        loan_number,
                        communication_start_date_time,
                        communication_end_date_time,
                        communication_direction,
                        communication_status,
                        communication_system,
                        communication_from_phone_number,
                        communication_to_phone_number,
                        vdn
                    FROM
                        :table_name;
                    WHERE 
                        communication_start_date_time >= CAST(:start_date; AS TIMESTAMP)
                        AND communication_start_date_time <= CAST(:end_date; AS TIMESTAMP)
                """
        if len(self.direction) > 0:
            query += """ AND trim(lower(communication_direction)) IN ({}) """.format(', '.join(f"'{direct.strip()}'" for direct in self.direction))
            
            query += """ 
                    ), 
                    toll_free_data AS
                    (
                    SELECT 
                        vdn,
                        toll_free_number
                    FROM 
                        rm_dp_conformed_access.communication_vdn
                    )
                    SELECT
                        call_communication.rktpersonid,
                        call_communication.communication_guid,
                        call_communication.loan_number,
                        call_communication.communication_start_date_time,
                        call_communication.communication_end_date_time,
                        call_communication.communication_direction,
                        call_communication.communication_status,
                        call_communication.communication_system,
                        call_communication.communication_from_phone_number,
                        call_communication.communication_to_phone_number,
                        call_communication.vdn,
                        toll_free_data.toll_free_number
                    FROM
                        call_communication
                    LEFT JOIN
                        toll_free_data ON 
                            call_communication.vdn = toll_free_data.vdn
                    LIMIT :limit; 
                """

        return query
    
class QuerySmsCommunication(QueryDataBase):
    
    """
    A child class which inherits all methods and properties from QueryDataBase used to represent communication data query

    Attributes
    ----------
    start_date : str
        Start date for the communication in 'yyyy-mm-dd' format 
    end_date : str
        End date for the communication in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving communication data. By default, this is set to 'communication_text'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    """
    
    def __init__(self, start_date, end_date, table_name='communication_text', limit='ALL', **kwargs):
        
        """
        super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """   
        super().__init__(start_date, end_date, table_name='communication_text', limit='ALL', **kwargs)
       
    def sql_instruction(self):
        
        """
        Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QuerySmsCommunication class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: 
                communication_direction, rktpersonid, loan_number, and sms status
        """
        
        query = """
                SELECT
                    loan_number,
                    rktpersonid,
                    actiondatetime AS actiondatetime_est,
                    sent,
                    delivered,
                    communication_direction,
                    istest
                FROM :table_name;
                WHERE CAST(communication_text.actiondatetime AS TIMESTAMP) >= CAST(:start_date; AS TIMESTAMP)
                    AND CAST(communication_text.actiondatetime AS TIMESTAMP) <= CAST(:end_date; AS TIMESTAMP)
                LIMIT :limit;
                """

        return query
    
class QueryEmailCommunication(QueryDataBase):
    
    """
    A child class which inherits all methods and properties from QueryDataBase used to represent communication data query

    Attributes
    ----------
    start_date : str
        Start date for the communication in 'yyyy-mm-dd' format 
    end_date : str
        End date for the communication in 'yyyy-mm-dd' format
    table_name : str
        Table name for retrieving communication data. By default, this is set to 'communication'
    limit : int
        Select a maximum number of records to retrieve. By default, this is set to 'ALL' unless modified
    """
    
    def __init__(self, start_date, end_date, table_name='communication', limit='ALL', **kwargs):
        
        """
        super() allows to access attributes in __init__ method in parent class QueryDataBase
        
        """   
        super().__init__(start_date, end_date, table_name='communication', limit='ALL', **kwargs)
       
    def sql_instruction(self):
        
        """
        Queries data based on provided parameters.

        Parameters
        ----------
        self : object
            QueryEmailCommunication class instance
        
        Returns
        -------
        query : str
            Parameterized SQL query for execution. Main fields include: 
                loan_number, communication_direction, rktpersonid, loan_number, and 
                email sent, opened, clicked, unsubscribed, and bounced event timestamps
        """
        
        query = """
                WITH email_sent_data AS 
                (
                SELECT
                    loan_number,
                    communication_to_party_guid,
                    communication_from_party_guid,
                    CASE
                        WHEN trim(lower(communication_direction)) = 'outgoing'
                        THEN communication_to_party_guid
                        ELSE communication_from_party_guid
                        END AS rktpersonid,
                    communication.communication_guid,
                    with_timezone(CAST(communication_start_date_time AS TIMESTAMP), 'US/Central') AT TIME ZONE 'US/Eastern' AS communication_start_date_time_est,
                    communication_direction,
                    communication_from_email_address,
                    communication_to_email_address,
                    email_sent_in_the_name_of
                FROM rm_dp_conformed_access.communication
                    LEFT JOIN rm_dp_conformed_access.communication_email ON
                        communication.communication_guid = communication_email.communication_guid
                    LEFT JOIN rm_dp_conformed_access.communication_mortgage ON
                        communication_mortgage.communication_guid = communication_email.communication_guid
                ),
                email_bounce_data AS 
                (
                SELECT
                    communication_email_bounce_event.communication_guid,
                    bouncesubcategory,
                    coalesce(try(date_parse(communication_email_bounce_event.email_bounce_event_date_time, '%Y-%m-%d %H:%i:%s.%f')),
                             try(date_parse(communication_email_bounce_event.email_bounce_event_date_time, '%Y-%m-%d %H:%i:%s'))
                            ) as email_bounce_event_date_time
                FROM rm_dp_conformed_access.communication_email_bounce_event
                ),
                email_open_data AS 
                (
                SELECT
                    communication_email_open_event.communication_guid,
                    with_timezone(CAST(email_open_event_date_time AS TIMESTAMP), 'US/Central') AT TIME ZONE 'US/Eastern' AS email_open_event_date_time_est
                FROM rm_dp_conformed_access.communication_email_open_event
                ),
                email_click_data AS 
                (
                SELECT
                    communication_email_click_event.communication_guid,
                    email_link_clicked,
                    with_timezone(CAST(email_click_event_date_time AS TIMESTAMP), 'US/Central') AT TIME ZONE 'US/Eastern' AS email_click_event_date_time_est
                FROM rm_dp_conformed_access.communication_email_click_event
                ),
                email_unsubscribe_data AS 
                (
                SELECT
                    communication_email_unsubscribe_event.communication_guid,
                    with_timezone(CAST(email_unsubscribe_event_date_time AS TIMESTAMP), 'US/Central') AT TIME ZONE 'US/Eastern' AS email_unsubscribe_event_date_time_est
                FROM rm_dp_conformed_access.communication_email_unsubscribe_event
                )
                SELECT
                    email_sent.*,
                    email_open.email_open_event_date_time_est,
                    email_click.email_click_event_date_time_est,
                    email_click.email_link_clicked,
                    email_unsubscribe.email_unsubscribe_event_date_time_est,
                    email_bounce.bouncesubcategory,
                    with_timezone(CAST(email_bounce.email_bounce_event_date_time AS TIMESTAMP), 'US/Central') AT TIME ZONE 'US/Eastern' AS email_bounce_event_date_time_est
                FROM email_sent_data AS email_sent
                LEFT JOIN email_bounce_data AS email_bounce ON
                    email_sent.communication_guid = email_bounce.communication_guid
                LEFT JOIN email_open_data AS email_open ON
                    email_sent.communication_guid = email_open.communication_guid
                LEFT JOIN email_click_data AS email_click ON
                    email_sent.communication_guid = email_click.communication_guid
                LEFT JOIN email_unsubscribe_data AS email_unsubscribe ON
                    email_sent.communication_guid = email_unsubscribe.communication_guid
                WHERE CAST(email_sent.communication_start_date_time_est AS TIMESTAMP) >= CAST(:start_date; AS TIMESTAMP)
                    AND CAST(email_sent.communication_start_date_time_est AS TIMESTAMP) <= CAST(:end_date; AS TIMESTAMP)
                    AND trim(lower(email_sent.communication_direction)) = 'outgoing'
                LIMIT :limit;
                """

        return query