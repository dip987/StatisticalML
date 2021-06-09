from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from exploration_helper_functions import *
from encoding import DataEncoder, OneHot
from sklearn.ensemble import GradientBoostingRegressor
from method_evaluation import evaluate_model
from missingdata import *
from sklearn.impute import KNNImputer


#################################################################################
# This file contains helpful code examples for trying out different imputation schemes and figuring out the impact
##################################################################################


data_path = Path(r'data/netflix_data.csv')
scheme = {'Genre': ReplaceWithValue(nan_string),
          'Tags': ReplaceWithValue(nan_string),
          'Languages': ReplaceWithValue(nan_string),
          'Series or Movie': ReplaceWithValue(nan_string),
          'Country Availability': ReplaceWithValue(nan_string),
          'Runtime': ReplaceWithValue(nan_string),
          'Director': ReplaceWithValue(nan_string),
          'Writer': ReplaceWithValue(nan_string),
          'Actors': ReplaceWithValue(nan_string),
          'Awards Received': ReplaceWithZero(),
          'Awards Nominated For': ReplaceWithZero(),
          'Release Date': ReplaceWithHighestFrequency(keep_all=False),
          'Netflix Release Date': ReplaceWithHighestFrequency(keep_all=False),
          'Production House': ReplaceWithValue(nan_string),
          'Title': DoNothing(),
          'IMDb Votes': DoNothing()}
categories_to_replace = ['Genre', 'Tags', 'Languages', 'Series or Movie', 'Country Availability', 'Runtime']
missing_handlers = [ReplaceWithValue(nan_string), ReplaceWithHighestFrequency()]

mean_matrix = np.zeros((len(categories_to_replace), len(missing_handlers)))
var_matrix = np.zeros_like(mean_matrix)
# Model
reg = GradientBoostingRegressor(random_state=42, n_estimators=120)

for i, category in enumerate(categories_to_replace):
    for j, missing_handler in enumerate(missing_handlers):
        missing_handler.reset()
        scheme.update({category: missing_handler})
        df = load_data(data_path)
        imputer = DataImputer(scheme=scheme)
        imputer.fit_transform(df)
        encoder = DataEncoder()
        x, y = encoder.fit_transform(dataframe=df)
        mean, var = evaluate_model(reg, x, y)
        mean_matrix[i, j] = mean
        var_matrix[i, j] = var




print('da')


