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
scheme = {
          'Release Date': DropColumn(),
          'Netflix Release Date': DropColumn(),
          }
categories_to_replace = ['Release Date']
encoding_schemes = [DropColumn(),
                    DateEncoding(format_string=r"%d %b %Y", month_encoding='one_hot'),
                    DateEncoding(format_string=r"%d %b %Y", month_encoding='ordinal')]

mean_matrix = np.zeros((len(categories_to_replace), len(encoding_schemes)))
var_matrix = np.zeros_like(mean_matrix)
# Model
reg = GradientBoostingRegressor(random_state=42, n_estimators=120)
df = load_data(data_path)
imputer = DataImputer()
imputer.fit_transform(df)

for i, category in enumerate(categories_to_replace):
    for j, encoding_scheme in enumerate(encoding_schemes):
        # Replace the corresponding column encoding scheme
        scheme.update({category: encoding_scheme})
        encoder = DataEncoder(scheme=scheme)
        x, y = encoder.fit_transform(dataframe=df)
        mean, var = evaluate_model(reg, x, y)
        mean_matrix[i, j] = mean
        var_matrix[i, j] = var
        scheme.update({category: DropColumn()})



