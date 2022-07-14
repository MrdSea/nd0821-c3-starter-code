# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
GBM classifier using slightly tuned parameters to predict census data.

## Intended Use
This model helps provide insight into income based on a number of factor

## Training Data
80% - nulls are removed and rest of data is clean

## Evaluation Data
20% split from original data

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Recall, Precision, and F-score are logged
prec:0.7787979966611018, recall:0.6078175895765472, fbeta:0.6827661909989022

## Ethical Considerations
Race and ethnicity may need further investigation on their bias.

## Caveats and Recommendations
Model can be improved through more tunning
