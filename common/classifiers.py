from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

import lightgbm as lgbm

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import *
from keras.callbacks import *


def train_predict_gb(df_X, df_y, df_X_test, model_config: dict):
    """
    Train a gradient boosting model with the specified hyper-parameters and return its predictions for the test data.
    """
    model_pair = train_gb(df_X, df_y, model_config)
    y_test_hat = predict_gb(model_pair, df_X_test, model_config)
    return y_test_hat


def train_gb(df_X, df_y, model_config: dict):
    """
    Train gradient boosting model with specified hyper-parameters and return the model and scaler if used.
    """
    # Double columns if required by shifts
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        max_shift = max(shifts)
        df_X = double_columns(df_X, shifts)
        df_X = df_X.iloc[max_shift:]
        df_y = df_y.iloc[max_shift:]

    # Scale data if specified
    is_scale = model_config.get("train", {}).get("is_scale", False)
    if is_scale:
        scaler = StandardScaler()
        scaler.fit(df_X)
        X_train = scaler.transform(df_X)
    else:
        scaler = None
        X_train = df_X.values

    y_train = df_y.values

    # Set model parameters
    params = model_config.get("params")
    lgbm_params = {
        'learning_rate': params.get("learning_rate"),
        'max_depth': params.get("max_depth"),
        "min_data_in_leaf": int(0.01 * len(df_X)),
        'num_leaves': 32,
        "lambda_l1": params.get("lambda_l1"),
        "lambda_l2": params.get("lambda_l2"),
        'is_unbalance': 'true',
        'boosting_type': 'gbdt',
        'objective': params.get("objective"),
        'metric': {'cross_entropy'},
        'verbose': 0,
    }

    model = lgbm.train(
        lgbm_params,
        train_set=lgbm.Dataset(X_train, y_train),
        num_boost_round=params.get("num_boost_round"),
    )

    return (model, scaler)


def predict_gb(models: tuple, df_X_test, model_config: dict):
    """
    Make predictions using the gradient boosting model and return predictions for the test data.
    """
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        df_X_test = double_columns(df_X_test, shifts)

    scaler = models[1]
    input_index = df_X_test.index
    if scaler:
        df_X_test = scaler.transform(df_X_test)
        df_X_test = pd.DataFrame(data=df_X_test, index=input_index)

    df_X_test_nonans = df_X_test.dropna()
    nonans_index = df_X_test_nonans.index

    y_test_hat_nonans = models[0].predict(df_X_test_nonans.values)
    y_test_hat_nonans = pd.Series(data=y_test_hat_nonans, index=nonans_index)

    df_ret = pd.DataFrame(index=input_index)
    df_ret["y_hat"] = y_test_hat_nonans
    sr_ret = df_ret["y_hat"]

    return sr_ret


def train_predict_nn(df_X, df_y, df_X_test, model_config: dict):
    """
    Train neural network with specified hyper-parameters and return predictions for the test data.
    """
    model_pair = train_nn(df_X, df_y, model_config)
    y_test_hat = predict_nn(model_pair, df_X_test, model_config)
    return y_test_hat


def train_nn(df_X, df_y, model_config: dict):
    """
    Train neural network model with specified hyper-parameters and return model and scaler if used.
    """
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        max_shift = max(shifts)
        df_X = double_columns(df_X, shifts)
        df_X = df_X.iloc[max_shift:]
        df_y = df_y.iloc[max_shift:]

    is_scale = model_config.get("train", {}).get("is_scale", True)
    if is_scale:
        scaler = StandardScaler()
        scaler.fit(df_X)
        X_train = scaler.transform(df_X)
    else:
        scaler = None
        X_train = df_X.values

    y_train = df_y.values

    params = model_config.get("params")
    layers = params.get("layers") or [X_train.shape[1] // 4]
    learning_rate = params.get("learning_rate")
    n_epochs = params.get("n_epochs")
    batch_size = params.get("bs")

    # Model architecture
    model = Sequential()
    for i, out_features in enumerate(layers):
        in_features = X_train.shape[1] if i == 0 else layers[i-1]
        model.add(Dense(out_features, activation='sigmoid', input_dim=in_features))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    es = EarlyStopping(monitor="loss", min_delta=0.0001, patience=3, verbose=0, mode='auto')

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=[es],
        verbose=1,
    )

    return (model, scaler)


def predict_nn(models: tuple, df_X_test, model_config: dict):
    """
    Make predictions using the neural network model and return predictions for the test data.
    """
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        df_X_test = double_columns(df_X_test, shifts)

    scaler = models[1]
    input_index = df_X_test.index
    if scaler:
        df_X_test = scaler.transform(df_X_test)
        df_X_test = pd.DataFrame(data=df_X_test, index=input_index)

    df_X_test_nonans = df_X_test.dropna()
    nonans_index = df_X_test_nonans.index

    tf.keras.backend.clear_session()

    y_test_hat_nonans = models[0].predict_on_batch(df_X_test_nonans.values)[:, 0]
    y_test_hat_nonans = pd.Series(data=y_test_hat_nonans, index=nonans_index)

    df_ret = pd.DataFrame(index=input_index)
    df_ret["y_hat"] = y_test_hat_nonans
    sr_ret = df_ret["y_hat"]

    return sr_ret


def train_predict_lc(df_X, df_y, df_X_test, model_config: dict):
    """
    Train logistic classifier with specified hyper-parameters and return predictions for the test data.
    """
    model_pair = train_lc(df_X, df_y, model_config)
    y_test_hat = predict_lc(model_pair, df_X_test, model_config)
    return y_test_hat


def train_lc(df_X, df_y, model_config: dict):
    """
    Train logistic classifier with specified hyper-parameters and return model and scaler if used.
    """
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        max_shift = max(shifts)
        df_X = double_columns(df_X, shifts)
        df_X = df_X.iloc[max_shift:]
        df_y = df_y.iloc[max_shift:]

    is_scale = model_config.get("train", {}).get("is_scale", True)
    if is_scale:
        scaler = StandardScaler()
        scaler.fit(df_X)
        X_train = scaler.transform(df_X)
    else:
        scaler = None
        X_train = df_X.values

    y_train = df_y.values

    args = model_config.get("params").copy()
    model = LogisticRegression(**args)

    model.fit(X_train, y_train)

    return (model, scaler)


def predict_lc(models: tuple, df_X_test, model_config: dict):
    """
    Make predictions using the logistic classifier model and return predictions for the test data.
    """
    shifts = model_config.get("train", {}).get("shifts", None)
    if shifts:
        df_X_test = double_columns(df_X_test, shifts)

    scaler = models[1]
    input_index = df_X_test.index
    if scaler:
        df_X_test = scaler.transform(df_X_test)
        df_X_test = pd.DataFrame(data=df_X_test, index=input_index)

    df_X_test_nonans = df_X_test.dropna()
    nonans_index = df_X_test_nonans.index

    y_test_hat_nonans = models[0].predict_proba(df_X_test_nonans.values)[:, 1]
    y_test_hat_nonans = pd.Series(data=y_test_hat_nonans, index=nonans_index)

    df_ret = pd.DataFrame(index=input_index)
    df_ret["y_hat"] = y
