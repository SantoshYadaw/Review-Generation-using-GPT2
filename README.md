Review Generation using GPT2
==============================

Author: Santosh Yadaw | [LinkedIn](https://www.linkedin.com/in/santosh-yadaw-b32025111/) | [Github](https://github.com/SantoshYadaw/)

Overview
------------

In this project, we aim to train a gpt2 open source model on amazon review dataset.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │


--------

Usage
------------
1. Build the necessary environment to run the code
```conda env create -f setup.yml```

2. Ensure your have downloaded the dataset from the repo and it is placed under data > raw folder.

3. Preprocess the raw data:
``` python -m src/data/make_dataset ```

4. Train the gpt2 model:
``` python src/models/train_model.py ```

5. Call inference using trained gpt2 model:
``` python src/models/predict_model ```


Exploratory Data Analysis
------------

Exploratory Data Analysis notebook can be found in [01_eda.ipynb](notebooks/01_eda.ipynb). Please refer to the notebook for the details.


Evaluation Metrics
------------

Model's Performance
------------

Future Work
------------

References
------------
