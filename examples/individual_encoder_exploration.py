from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from exploration_helper_functions import *
from encoding import *
from sklearn.ensemble import GradientBoostingRegressor
from method_evaluation import evaluate_model
from missingdata import *
from sklearn.impute import KNNImputer

#################################################################################
# This file contains helpful code examples for trying out different imputation schemes and figuring out the impact
##################################################################################


data_path = Path(r'data/netflix_data.csv')
scheme = {'Genre': DropColumn(),
          'Tags': DropColumn(),
          'Languages': DropColumn(),
          'Series or Movie': DropColumn(),
          'Country Availability': DropColumn(),
          'Runtime': DropColumn(),
          'Director': DropColumn(),
          'Writer': DropColumn(),
          'Actors': DropColumn(),
          'Awards Received': DropColumn(),
          'Awards Nominated For': DropColumn(),
          'Release Date': DropColumn(),
          'Netflix Release Date': DropColumn(),
          'Production House': DropColumn(),
          }
categories_to_replace = ['Genre', 'Tags', 'Languages', 'Series or Movie', 'Country Availability', 'Runtime', 'Director',
                         'Writer', 'Actors']
encoding_schemes = [KeepTopN(1), KeepTopN(5), KeepTopN(10), KeepTopN(15), KeepTopN(20), KeepTopN(25)]
encoding_schemes3 = [TargetPriorityNEncoding(1), TargetPriorityNEncoding(5), TargetPriorityNEncoding(10),
                     TargetPriorityNEncoding(15), TargetPriorityNEncoding(20), TargetPriorityNEncoding(25)]

mean_matrix = np.zeros((len(categories_to_replace), len(encoding_schemes)))
var_matrix = np.zeros_like(mean_matrix)
# Model
reg = GradientBoostingRegressor(random_state=42, n_estimators=120)
df = load_data(data_path)
imputer = DataImputer()
imputer.fit_transform(df)

encoding_schemes2 = [KeepTopN(1), KeepTopN(5), KeepTopN(10)]

mean_matrix_2 = np.zeros((len(categories_to_replace), len(encoding_schemes2)))
var_matrix_2 = np.zeros_like(mean_matrix_2)

for i, category in enumerate(categories_to_replace):
    for j, encoding_scheme in enumerate(encoding_schemes):
        # Replace the corresponding column encoding scheme
        scheme.update({category: encoding_scheme})
        encoder = DataEncoder(scheme=scheme)
        x, y = encoder.fit_transform(dataframe=df)
        mean, var = evaluate_model(reg, x, y)
        mean_matrix_2[i, j] = mean
        var_matrix_2[i, j] = var
        scheme.update({category: DropColumn()})

fig, ax = plt.subplots(len(categories_to_replace), 1, sharex='col')



