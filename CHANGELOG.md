# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.5] - 2023-10-26
### Updated
- Modified ownership to reflect current status

## [1.0.4] - 2023-03-25

### Updated
- Fixed potential issue with funnel plot in plotter

### Added
- Capability designed to provide interoperability between Presto SQL, Spark, and other dialects
- Query optimization method to improve performance of user-defined queries and traverse expression trees
- Added call communication SQL instructions and associated transformation functions

## [1.0.3] - 2023-01-26

### Updated
- Fixed potential issue with 'string' data type in label encoder

### Added
- Exception handling for color map bug in plotly for plotting conversion funnel
- Data Type conversion has verbose option added as an optional argument
- Added data query to import the following features
    - Email
    - Soft credit
    - Employment status
    - Bankruptcy
    - Country
    - Region
    - City
    - Home value
    - Mortgage balance
    - Refinance reason
    - Second mortgage

### Removed
- External Dependencies no longer needed are removed

## [1.0.2] - 2022-12-16

### Updated
- Fixed inconsistencies in Query Label 'QueryAdobeLeadformInput'
- Renamed library modules and submodules

### Added
- Logger for all modules and submodules
- Implementation of Tree Data Structure for Project Overview
- Plots for Univariate and Bivariate Histograms, KDE, Bar plots, Correlation Matrix, Cramer's V Association
- Visualization for data segments based on pre-defined partitioning strategies
- Confusion Matrix plot
- ROC AUC and Precision-Recall (PR) Curve with Thresholds
- Convenience functions to convert data types
- Convenience functions to download and delete data in AWS s3 bucket
- Highlight Pandas DataFrames
- Data Profiler
- Outlier detection and removal capability
- Data Cleansing and Transformation capability
- Data Partitioning algorithms (supports 'TimeSeriesSplit' and 'KFold' only)
- Label encoding for categorical (dichotomous, nominal, ordinal) variables 
- Hyperparameter Tuning Job Progress plot
- ML model training for deployment (XGBoost support only)
- Experiment Tracking and Model Registration on Mlflow

### Removed
- Preprocessing function due to duplication in Utility functions
- import_export function due to duplication in Utility functions
- plot_data file due to duplication in Visualization functions

## [1.0.1] - 2022-10-21

### Updated
- requirements.txt file to add boto3 and awswrangler packages that were being referenced in test class
- added some print for logging version to console when tests are executed.
- VERSION updated to have semantic version format to bump up a minor version as well.

### Added
- CHANGELOG file

### Removed
- code from all init files from submodules, because the src/ module get VERSION accessible across