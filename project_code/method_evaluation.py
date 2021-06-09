from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import tensorflow as tf


def evaluate_model(model, x, y, k=5, fit_params=None):
    """
    Evaluates the given model using KFold CV and returns the mean and variance of MSE over all the
    cross-folds

    :param model: regression model to use (For this project using XGBoost)
    :param x: Feature Vector X after missing data imputation AND encoding!
    :param y: Result Variable y
    :param k: K-Fold Cross-Validation. By default 5
    :param fit_params: Fitting params to pass into the model
    :return: Mean, Variance
    """

    scores = -1 * cross_val_score(model, x, y, cv=k, scoring='neg_mean_absolute_error')
    mean = np.average(scores)
    var = np.var(scores)
    return mean, var


def evaluate_model_tf(model: tf.keras.Sequential, x: np.ndarray, y: np.ndarray, epochs: int = 60,
                      cv: int = 4, batch_size: int = 32):
    mse_scores = []
    splits_X = np.split(x, cv)
    splits_y = np.split(y, cv)
    for trial in range(cv):
        X_test = splits_X[trial]
        y_test = splits_y[trial]
        train_indices = list(range(cv))
        train_indices.pop(trial)
        X_train = np.vstack([splits_X[index] for index in train_indices])
        y_train = np.vstack([splits_y[index].reshape(-1, 1) for index in train_indices])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
        error = model.evaluate(X_test, y_test)
        mse_scores.append(error)
        tf.keras.backend.clear_session()
    return mse_scores
