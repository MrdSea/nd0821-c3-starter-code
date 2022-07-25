import pandas as pd
import numpy as np
import sys
import pickle
from sklearn.model_selection import train_test_split
import os
import pytest

sys.path.append("..")
from model_train.base.data import process_data
from model_train.base import model


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

def test_process_data(data,cat_features):
    X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True)
    assert len(X) == len(y)

def test_train_output():
    assert os.path.isfile("../model/encoder.pkl")
    assert os.path.isfile("../model/lb.pkl")
    assert os.path.isfile("../model/model.pkl")