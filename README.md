Synopsys Project 2016-2017
==============================
### Leveraging Deep Learning to Derive _De Novo_ Epigenetic Mechanisms from _PPARGC1A_ to Account for Missing Heritability in Type II Diabetes Mellitus

### Data / Statistical Analysis Jupyter Notebooks
Until Github fixes notebook renderings, please visit the following links (same as notebooks/)
https://nbviewer.jupyter.org/github/minhoolee/Synopsys-Project-2017/blob/master/notebooks/0.1-mhl-data-analysis.ipynb  
https://nbviewer.jupyter.org/github/minhoolee/Synopsys-Project-2017/blob/master/notebooks/0.1-mhl-model-predictions.ipynb  
https://nbviewer.jupyter.org/github/minhoolee/Synopsys-Project-2017/blob/master/notebooks/0.2-mhl-model-predictions.ipynb  

### Synopsys Competition Tri-Fold
<img src="https://cloud.githubusercontent.com/assets/10465228/24325410/a0631506-1155-11e7-8f5e-756332d353d1.jpg"/>

The focus of the project was in using deep learning to predict novel epigenetic 
mechanisms like __DNase I sites__, __histone modifications__, and 
__transcription factor binding sites__ from raw genomic sequences. Type II 
diabetes (T2D) is a common disease that affects millions of people each year, 
but as of today, only around 10% of its heritability has been explained. 
Researchers speculate that this is because epigenetics is heavily involved, so 
my project was designed to interpret millions of samples and hundreds of 
epigenetic regulators to be able understand the combinatorial effects of these 
epigenetic mechanisms.

I conducted this independent research project for the Synopsys science fair as 
a high school junior. In order to train my models, I built my own custom PC [(see
specs here)](https://pcpartpicker.com/user/minhoolee/saved/GZskLk). I would like to 
thank my mentor, Renee Fallon, in providing me biology textbooks and general advice.

### Custom Built PC 
<img src="https://cloud.githubusercontent.com/assets/10465228/24325200/e7be4be6-1150-11e7-82ef-5f7c4ba73ca3.JPG"/>

## Steps for reproducing results

### Step 1. Get data
Download [processed data from DeepSEA](http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz) 
and move them to `data/processed/`
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

### Step 2. Create model
Create a method in src/models/create_models.py that constructs a Keras model 
(sequential, functional, etc.) and then returns it.

### Step 3. Train model
Run `make train MODEL_FUNC='<method from step 2>' MODEL_NAME='<some unique identifier>'`

### Step 4. Test model and generate predictions
Run `make test MODEL_FUNC='<same as from step 3>' MODEL_NAME='<same as from step 3>'`

### Step 5. Generating performance (ROC/PR, stdev, etc.) scores and visualizations
See notebooks/ and run the code after "Execute the following" headers. Make 
sure to run them with the Theano backend for Keras because the models were all 
trained on Theano.

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
