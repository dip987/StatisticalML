# Predicting The Reach of A Show
### A Project by Rishad Raiyan

## Report
I would suggest reading the report first and then moving on to the notebooks. The notebooks contain a bit more explanation than the report. The project codes are properly documented and you can go through them with your IDE of choice. Just open the StatisticalML folder as a project

## Data
The data is stored inside 'data/netflix_data.csv'. The notebook '/data_exploration.ipynb' also gives a good insight on the data by using some processing. Furhter exploration is reserved for the example notebooks inside 'notebooks/'

## Pipeline
The pipeline codes are stored inside 'project_code/'. The 'exploration_helper_function.py' contains functions for loading data and splitting up entries. The 'missing.py' and 'encoding.py' consists of *Missing Data Handlers* and *Data Encoders*. They contain different types of handlers and encoders and a wrapper class to unify them all. The 'method_evaluation.py' functions to evaluate the performance of a given model. An example for how to use the pipeline is provided in 'notebooks/notebook_example.ipynb'.

## Examples
Example codes for trying out different encoding, imputation and different models are given along with proper explanations inside the jupyter notebook files stored inside 'notebooks/'. Most of the code from those notebooks are also stored as python files inside 'examples/'. These notebooks also demonstrate how the different parts of this pipeline work and why speicifc schemes were selected for the final evaluation.

## Extras
This project uses numpy, sklearn, pandas and tensorflow 2.0 libraries. A complete list of tall the libraries installed on my system are given in the text file environment.txt. There are however a lot of packages installed on my environment so it might not be super useful
