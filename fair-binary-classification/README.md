# fair-binary-classification
Fairness-aware binary classification for the [Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult)

## Contents
This repository contains two jupyter notebooks for predictive modeling of the adult dataset:
 - `adult_basic.ipynb` includes a fairness-unaware approach using [pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org/stable/)
 - `adult_fair.ipynb` utilizes [AIF360](https://aif360.mybluemix.net/fairness) toolset and its metrics and bias mitigation techniques to reduce bias w.r.t. sensitive attributes present in the dataset

Both notebooks provide detailed information about the analysis and the algorithms that are utilized within.
