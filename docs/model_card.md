# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf



## Model Details

In this experiment, I've used Random forest classifier for prediction. 
Default parameters are used for the model.

## Intended Use

The model is intended to be used for predicting the income of a person based on the census data.

## Training Data

The source of the data is from `https://archive.ics.uci.edu/ml/datasets/census+income`. 
80% of the data is used for training.

## Evaluation Data

The source of the data is from `https://archive.ics.uci.edu/ml/datasets/census+income`. 
20% of the data is used for evaluation.


## Metrics

Metrics are used to judge the performance of a model. In this project, we use the following metrics:

* Precision
* Recall
* Fbeta score


## Ethical Considerations

This project is for educational purpose only. The model is not intended to be used in production.
This model may be biased towards certain groups of people.

## Caveats and Recommendations

The model is not intended to be used in production. It is only for educational purpose.
