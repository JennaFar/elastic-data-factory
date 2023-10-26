import os
from setuptools import find_packages, setup

with open(os.path.join("elasticdatafactory", "version.txt"), "r") as version_file:
    version = version_file.read().strip()

with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

with open("LICENSE", "r", encoding="utf-8") as license_file:
    license_text = license_file.read()

setup(
    name='elastic-data-factory',
    package_dir={'': "./"},
    packages=find_packages(where="./", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version=version,
    description='Elastic Data Factory for Feature Discovery and Visualization',
    author='Jenna Far',
    author_email="jennafar@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=license_text,
    platform=any,
    include_package_data=True,
    url="https://github.com/JennaFar/elastic-data-factory",
    project_urls={
        "Bug Tracker": "https://github.com/JennaFar/elastic-data-factoryissues",
        "Changelog": "https://github.com/JennaFar/elastic-data-factory/blob/main/CHANGELOG.md",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        'treelib>=1.6.1',
        'click>=8.1.3',
        'awswrangler==2.20.1',
        'boto3>=1.26.24',
        'sagemaker>=2.120.0',
        'pandas>=1.4.4',
        'numpy>=1.23.3',
        'scikit-learn>=1.2.0',
        'scipy>=1.9.1',
        'tqdm>=4.64.1',
        'matplotlib>=3.6.2',
        'seaborn>=0.12.1',
        'plotly>=5.11.0',
        'statsmodels>=0.13.5',
        'mlflow==2.2.2',
        'mlflow_connect>=0.4.0',
        'xgboost>=1.2.0',
        'sqlglot>=11.4.1'
    ],
    extras_require={
        'interactive': ['notebook>=6.1.5', 'jupyter>=1.0.0'],
        'docs': ['sphinx>=3.5'],
    },
    python_requires=">=3.9",
)

