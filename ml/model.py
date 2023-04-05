import joblib
import os
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

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

    # train a random forest model and return it
    model = RandomForestClassifier()
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

def save_model(model, encoder, lb):
    """ Save the model to a file.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    encoder : ???
        Encoder used to transform the data.
    lb : ???
        Label encoder used to transform the labels.
    """
    if not os.path.exists("model"):
        os.mkdir("model")
    joblib.dump(model, "model/model.joblib")
    joblib.dump(encoder, "model/encoder.joblib")
    joblib.dump(lb, "model/lb.joblib")

def load_model():
    """ Load the model from a file.

    Returns
    -------
    model : ???
        Trained machine learning model.
    encoder : ???
        Encoder used to transform the data.
    lb : ???
        Label encoder used to transform the labels.
    """
    model = joblib.load("model/model.joblib")
    encoder = joblib.load("model/encoder.joblib")
    lb = joblib.load("model/lb.joblib")
    return model, encoder, lb