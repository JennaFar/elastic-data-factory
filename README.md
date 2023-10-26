# Elastic-Data-Factory
<hr>

### Project Description
This library provides a consolidated resource of data queries, data profiling tools, data processing and analysis, feature engineering and visualization, model training through Sagemaker and evaluation functions, experiment tracking, and model registering capabilities through MLflow. All queries are stored as objects with appropriate labels for convenient discovery and sharing. This library includes a wrapper to execute queries based on user-defined parameters in query registry, and allows the ability to edit query to user's needs before returning it to wrapper for execution. There are several utility and helper functions included in this build to interact with built-in capabilities and MLflow/AWS services during a Data Science lifecycle.

In order to query data through the wrapper, use the following arguments:

```
Usage: make_dataset.py start_date end_date query_label
                       database workgroup limit
Try 'make_dataset.py --help' for help.

Example:

python make_dataset.py 2021-11-28 2021-11-28 QueryMajorMilestone 
                       rm_northstar_raw_processed_access rcd-datascientist ALL
```

Project Organization
------------

    Elastic-Data-Factory
    ├── CHANGELOG.md
    ├── MANIFEST.in
    ├── Makefile                        <- Makefile with commands like `make data` or `make train`
    ├── README.md                       <- The top-level README for developers using this project.
    ├── docs                            <- A default Sphinx project; see sphinx-doc.org for details
    │   ├── CODEOWNERS
    │   ├── Makefile
    │   ├── commands.rst
    │   ├── conf.py
    │   ├── getting-started.rst
    │   ├── index.md
    │   ├── index.rst
    │   ├── make.bat
    │   └── notebooks.rst
    ├── elasticdatafactory              <- Project source code containing Python modules
    │   ├── VERSION                     <- Project source code version
    │   ├── data                        <- Scripts to download or generate data
    │   │   ├── field_query_mapping.json
    │   │   ├── make_dataset.py
    │   │   ├── query_db_map.json
    │   │   └── query_registry.py
    │   ├── features                    <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   ├── plots                       <- Scripts to create exploratory and results oriented visualizations
    │   │   └── plotter.py
    │   ├── trainers                    <- Scripts to train models and then use trained models to make
    │   │   ├── predict_model.py
    │   │   ├── train_deploy.py
    │   │   └── train_model.py
    │   └── utilities                   <- Utility functions for data manipulation and to interact with AWS and MLflow Services
    │       ├── helper.py
    │       └── utility.py
    ├── logger.log
    ├── make.bat
    ├── mkdocs.yml
    ├── models                          <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks                       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                                  the creator's initials, and a short `-` delimited description, e.g.
    │   │                                  `1.0-jqp-initial-data-exploration`.
    │   ├── data_wrangling_jenna.ipynb
    │   ├── develop_test_EDF.ipynb
    │   ├── logger.log
    │   ├── long_time_frame_to_buy.ipynb
    │   └── query_web_events_wwen.ipynb
    ├── pyproject.toml
    ├── references
    ├── reports                         <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                     <- Generated graphics and figures to be used in reporting
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g.
    │                                      generated with `pip freeze > requirements.txt`
    ├── scripts
    │   └── deploy-ghpages.sh
    ├── setup.cfg
    ├── setup.py                        <- makes project pip installable (pip install -e .) so src can be imported
    ├── source
    │   ├── conf.py
    │   └── index.rst
    ├── test_environment.py
    ├── tests
    │   └── unit_tests
    │       ├── query_db_map.json
    │       └── test_queryregistry.py
    └── tox.ini                         <- tox file with settings for running tox; see tox.readthedocs.io

--------
