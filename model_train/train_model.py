# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.

import pandas as pd
from base import data
from base import model
import pickle

# Add code to load in the data.

df = pd.read_csv('../data/censusclean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# Train a model.
output_model = model.train_model(X_train, y_train)

#save the model
with open('../model/model.pkl', 'wb') as f:
    pickle.dump(output_model, f)

# Save the encoder and label binarizer.
with open('../model/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('../model/lb.pkl', 'wb') as f:
    pickle.dump(lb, f)

#train and test inference
train_preds = model.inference(output_model, X_test)
test_preds = model.inference(output_model, X_train)

#model metrics 
precision, recall, fbeta = model.compute_model_metrics(y_test,test_preds)
print (f"prec:{precision}, recall:{recall}, fbeta:{fbeta}")