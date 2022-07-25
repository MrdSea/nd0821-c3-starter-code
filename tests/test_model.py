import sys
sys.path.append("..")
from model_train.base.data import process_data
import pandas as pd
import os
import pytest

@pytest.fixture()
def data():
    df = pd.read_csv('../data/censusclean.csv')
    return df


@pytest.fixture()
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]


def test_data_size(data):
    assert len(data) > 1000


def test_process_data(data, cat_features):
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True)
    assert len(X) == len(y)


def test_train_output():
    assert os.path.isfile("../model/encoder.pkl")
    assert os.path.isfile("../model/lb.pkl")
    assert os.path.isfile("../model/model.pkl")
