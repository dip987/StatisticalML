from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from missingdata import DataImputer
from exploration_helper_functions import *
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from encoding import DataEncoder, OneHot, TargetPriorityNEncoding
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from method_evaluation import evaluate_model
from missingdata import *
from exploration_helper_functions import split_entries_with_target_mean
import tensorflow as tf

data_path = Path(r'data/netflix_data.csv')
df = load_data(data_path)

imputer = DataImputer()
imputer.fit_transform(df)
encoder = DataEncoder()
x, y = encoder.fit_transform(dataframe=df)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(16,  activation='relu'),
                                    tf.keras.layers.Dense(16,  activation='relu'),
                                    tf.keras.layers.Dense(8,  activation='relu'),
                                    tf.keras.layers.Dense(8,  activation='relu'),
                                    tf.keras.layers.Dense(4,  activation='relu'),
                                    tf.keras.layers.Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(x.toarray(), y, epochs=30, validation_split=0.2)
# evaluate_model(model, x, y)
