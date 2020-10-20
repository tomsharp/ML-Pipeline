# Predicting Loan Defaults

## Background
The purpose of this repository is to create a robust, automated Machine Learning (ML) pipeline. While the current settings of the repo are focused on a the Credit Risk Assesment dataset, the code is written in a modular format so that the dataset and models used can easily be changed. 
<br><br>

## Contents
**src/config.py**
<br>Configurable information about the dataset. Here you can change which dataset the code points to, which columns of that dataset to bring in, the acceptable range that the data should lie in (numeric columns), the categories that the data should contain (categorical/ordinal data), and the specific encodings to use in feature engineering.

**src/preprocess.py**
<br>Wrapper functions for the sepcific steps of preparing and splitting the data before modeling occurs. 
For data preparation, data quality (DQ) checks are used at import to ensure the data that the model will be training on does not contain any unspecified values (via the *config.py* file). The data is then encoded according to the *config.py* file.
For data splitting, the data is split in standard training and testing sets. An optional parameter to balance the classes is set as default, which allows for a resampling procedure to balance the positive and negative classes in the training and test set. 

**src/model.py**
<br>Provides automated gridsearch for several models at once. A user must simply update the `model` dictionary with their desired SKLearn models and parameters for grid search,  and run the script (the user may also update the scoring criteria for the gridsearch, if desired). The script will automatically gridsearch across the parameter grid for each model and find the best parameters according to the scoring criteria. The best parameters will then be used to fit the data again (if the `test_thresholds` parameter is set to true, the script will evaluate the model at different thresholds for predict_proba). The results from each of these models is saved in a csv file in the *modeling/* folder. 

## Quick Start - How to use this repo
### Set up conda environment
```
$ conda create --name ml_pipeline --y
$ conda activate ml_pipeline
$ pip install -r requirements.txt
```
### Run automated modeling
```
$ python src/model.py
```