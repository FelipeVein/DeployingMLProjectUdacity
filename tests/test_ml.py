
from ml.model import train_model, inference, compute_model_metrics, save_model, load_model
from ml.data import process_data
from ml.pipelines import training_pipeline, prediction_pipeline, load_data
import ml.constants as constants
import os
import sklearn
import pytest
import numpy as np
import pandas as pd


def test_constants():
    assert os.path.exists(constants.DATA_FILE)
    assert constants.LABEL == 'salary'
    assert len(constants.CATEGORICAL_FEATURES) == 8


@pytest.fixture(scope="module")
def data():
    train, test = load_data()
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train.columns) == 15
    return train, test

def test_process_data(data):
    train, test = data
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=constants.CATEGORICAL_FEATURES,
        label=constants.LABEL
    )
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=constants.CATEGORICAL_FEATURES,
        label=constants.LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder)
    assert isinstance(lb, sklearn.preprocessing._label.LabelBinarizer)

# i want to model to be send to another test function

@pytest.fixture(scope="module")
def test_train_model(data):
    train, test = data
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=constants.CATEGORICAL_FEATURES,
        label=constants.LABEL
    )
    model = train_model(X_train, y_train)
    assert isinstance(model, sklearn.ensemble._forest.RandomForestClassifier)
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder)
    assert isinstance(lb, sklearn.preprocessing._label.LabelBinarizer)
    return model, encoder, lb

def test_save_and_load_model(test_train_model):
    model, encoder, lb = test_train_model
    save_model(model, encoder, lb)
    model, encoder, lb = load_model()
    assert isinstance(model, sklearn.ensemble._forest.RandomForestClassifier)
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder)
    assert isinstance(lb, sklearn.preprocessing._label.LabelBinarizer)

def test_training_pipeline():
    output = training_pipeline()
    assert isinstance(output['precision'], float)
    assert isinstance(output['recall'], float)
    assert isinstance(output['fbeta'], float)
    assert isinstance(output['model'], sklearn.ensemble._forest.RandomForestClassifier)
    assert isinstance(output['encoder'], sklearn.preprocessing._encoders.OneHotEncoder)
    assert isinstance(output['lb'], sklearn.preprocessing._label.LabelBinarizer)

def test_prediction_pipeline(data):
    train, test = data
    preds = prediction_pipeline(test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(test)
