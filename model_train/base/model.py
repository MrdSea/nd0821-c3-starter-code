from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier as gbm
from .data import process_data
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = gbm(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_slice_metrics(df,cat_features,model,encoder,lb):
    output = []
    for column in df.columns:
        for val in df[column].unique():
            df_slice = df[df[column] == val]
            X, y, _, _ = process_data(df_slice, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
            preds = inference(model,X)
            precision, recall, fbeta = compute_model_metrics(y,preds)
            output.append([column, val, precision, recall, fbeta])
    slice_output = pd.DataFrame(output, columns=['column', 'value', 'precision', 'recall', 'fbeta'])
    slice_output.to_csv('slice_output.txt', index=False)
