Synopsys Project 2016-2017
==============================

Leveraging Deep Learning to Derive De Novo Epigenetic Mechanisms from PPARGC1A to Account for Missing Heritability in Type II Diabetes Mellitus

Download [processed data from DeepSEA](http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz) and move them to `data/processed/`
Data is processed in the following manner:

> Data on histone modifications, DNase I hypersensitive sites, and 
transcription factor binding sites is collected from ChIP-seq and DNase-seq 
methods. This data entails 919 ChIP-seq and DNase-seq peaks from processed 
ENCODE and Roadmap Epigenomics data releases for GRCh37. This data is 
publically available to download and has been processed by the researchers of 
the DeepSEA framework (Zhou). The input is encoded in a 1000 x 4 binary matrix, 
with the columns corresponding to A, T, G, and C. The rows corresponds to the 
number of bp (1kbp) in a single bin that will serve as the input for a single 
neuron. These 1000 bp regions are centered around 200 bp sequences that contain 
at least one transcription factor site (400 bp sequence paddings for genome 
sequence context). The data is split into test, train, and validation sets, and 
the sets are separated based off of chromosomes in order to ensure that the 
model can be tested for high bias.

Project Organization
------------

    ├── LICENSE
    ├── Makefile                  <- Makefile with commands like `make data` or `make train`
    ├── README.md                 <- The top-level README for developers using this project.
    ├── data
    │   ├── external              <- Data from third party sources.
    │   ├── interim               <- Intermediate data that has been transformed.
    │   ├── processed             <- The final, canonical data sets for modeling.
    │   └── raw                   <- The original, immutable data dump.
    │
    ├── docs                      <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                    <- Trained and serialized models, model predictions, or model summaries
    │   ├── csv                   <- CSV logs of epoch and batch runs
    │   ├── json                  <- JSON representation of the models
    │   ├── predictions           <- Predictions generated the train models and their best weights
    │   ├── weights               <- Best weights for the models
    │   └── yaml                  <- YAML representation of the models
    │
    ├── notebooks                 <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                the creator's initials, and a short `-` delimited description, e.g.
    │                                `1.0-jqp-initial-data-exploration`.
    │
    ├── references                <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                   <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures               <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt          <- The requirements file for reproducing the analysis environment, e.g.
    │                                generated with `pip freeze > requirements.txt`
    │
    ├── src                       <- Source code for use in this project.
    │   ├── __init__.py           <- Makes src a Python module
    │   │
    │   ├── data                  <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features              <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── logging               <- Scripts to improve python logging
    │   │   └── log_utils.py
    │   │
    │   ├── models                <- Scripts to train and test models and then use trained models to make
    │   │   │                         predictions
    │   │   ├── create_models.py  <- Script to create a keras model and return it to train_model.py
    │   │   ├── predict_model.py
    │   │   ├── test_model.py
    │   │   └── train_model.py
    │   │
    │   │── unit_tests            <- Scripts to test each unit of the other scripts
    │   │
    │   └── visualization         <- Scripts to create exploratory and results oriented visualizations
    │       ├── plot_train_valid.py
    │       ├── stats.py
    │       └── visualize.py
    │
    └── tox.ini                   <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
