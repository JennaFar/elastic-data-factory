import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import awswrangler as wr
import boto3
import json
from pandas.testing import assert_frame_equal
from elasticdatafactory.data import make_dataset
from elasticdatafactory.utilities.utility import load_json_from_s3, populate_modules, import_json

s3 = boto3.client('s3')
bucket_name = 'sagemaker-us-east-2-534295958235'
prefix = 'rcd/queryregistry_unittest/'

object_dict = populate_modules(module_name='data.query_registry')
query_db_mapping = import_json(r'query_db_map.json')


class Test_TestQueryRegistry(unittest.TestCase):

    def test_query_output_dataframe(self):
        additional_args = {
            'QueryAdobeEvents': {'event_codes': ['206']},
            'QueryAdobeLeadformInput': {'site_sections': ['ql lander']},
            'QueryMajorMilestone': {'milestones': []}
        }
        query_sort_keys = {
            'QueryAccountCreate': ['recordid'],
            'QueryAdobeEvents': ['uniquevisitkey', 'date_time'],
            'QueryAdobeLeadformInput': ['uniquevisitkey', 'mcvisid_visitorid'],
            'QueryAdobeVisitId': ['uniquevisitkey'],
            'QueryAdobeVisitInfo': ['uniquevisitkey', 'mcvisid_visitorid'],
            'QueryExperimentVisitorData': ['mcvisid_visitorid', 'rowNumber'],
            'QueryLeadSubmission': ['recordid'],
            'QueryLoanIdentifier': ['loanidentifierdimsk'],
            'QueryMajorMilestone': ['loanidentifierdimsk', 'estdatetime'],
            'QueryPreloanId': ['recordid'],
            'QueryPreloanPurpose': ['recordid']
        }

        for object_name, obj in object_dict.items():
            if object_name in query_sort_keys:
                args = {'query_label': object_name,
                        'start_date': '2022-07-18',
                        'end_date': '2022-07-18',
                        'database': query_db_mapping[object_name][0],
                        'limit': 'ALL',
                        'workgroup': 'rcd-datascientist',
                        'output_filepath': None}
                if object_name in additional_args:
                    args.update(additional_args[object_name])
                output_df = make_dataset.main.callback(**args)
                output_df.sort_values(by=query_sort_keys[object_name], inplace=True)

                test_file_name = object_name.lower()
                cat_output = output_df.select_dtypes(exclude=['number', 'datetime', 'timedelta'])
                if cat_output.shape[1] > 0:
                    output_json = cat_output.describe().to_json()
                    test_json = load_json_from_s3(test_file_name + '_cat.json', s3, bucket_name, prefix)
                    self.assertEqual(output_json, test_json)

                num_output = output_df.select_dtypes(include=['number'], exclude=['datetime', 'timedelta'])
                if num_output.shape[1] > 0:
                    output_json = num_output.describe().to_json()
                    test_json = load_json_from_s3(test_file_name + '_num.json', s3, bucket_name, prefix)
                    self.assertEqual(output_json, test_json)

    def test_undefined_query_argument(self):

        for object_name, obj in object_dict.items():
            if object_name != 'QueryDataBase':
                args = {'query_label': object_name,
                        'start_date': '2022-06-21',
                        'end_date': '2022-06-21',
                        'database': query_db_mapping[object_name][0],
                        'limit': 1,
                        'workgroup': 'rcd-datascientist',
                        'output_filepath': None,
                        'unknown_arg': 1}

                output_df = make_dataset.main.callback(**args)
                self.assertEqual(len(output_df), 1)


if __name__ == '__main__':
    print("App __version__: ", __version__)
    unittest.main()
