import pandas as pd
from sklearn.model_selection import train_test_split
import sys

from ml.data import process_data, slice_data
from ml.constants import CATEGORICAL_FEATURES, LABEL, DATA_FILE
from ml.model import compute_model_metrics, inference, save_model, train_model, load_model

def training_pipeline():
    """
    Trains a machine learning model and saves it.
    """
    # Load data
    df_train, df_test = load_data()
    # Process data
    X_train, y_train, encoder, lb = process_data(
        df_train, categorical_features=CATEGORICAL_FEATURES, label=LABEL
    )
    X_test, y_test, encoder, lb = process_data(
        df_test,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train model
    model = train_model(X_train, y_train)
    # Validate model
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision}, Recall: {recall}, F1: {fbeta}")
    # Save model
    save_model(model, encoder, lb)
    
    return {"precision": precision, "recall": recall, "fbeta": fbeta,
            "model": model, "encoder": encoder, "lb": lb, 
            "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def load_data():
    """ Load the data from the data source.
    Returns
    -------
    train : pd.DataFrame
        Training data.
    test : pd.DataFrame
        Test data.
    """
    # Load data from data source
    df = pd.read_csv(DATA_FILE)
    # Split data into train and test
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test


def prediction_pipeline(X, model=None, encoder=None, lb=None, process=True):
    """ Run the prediction pipeline.
    Parameters
    ----------
    X : pd.DataFrame
        Data to be used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    if model is None:
        # Load model
        model, encoder, lb = load_model()
    if process:
        # Process data
        X, _, _, _ = process_data(
            X,
            categorical_features=CATEGORICAL_FEATURES,
            label=LABEL,
            training=False,
            encoder=encoder,
            lb=lb,
        )
    # Run inference
    preds = inference(model, X)
    return preds

def slice_performance():
    """ 
    Run the prediction pipeline on slices of data.
    Each slice is a part of the test data that has the same value for a specific categorical feature.
    Saves the performance of the model on each slice to a file.
    Parameters
    ----------
    X : pd.DataFrame
        Data to be used for prediction.
    Returns
    -------
    None
    """
    # Load model
    model, encoder, lb = load_model()
    # Load data
    df_train, df_test = load_data()
    # Save printout to file
    with open("slice_performance.txt", "w") as f:
        sys.stdout = f
        # Slice data by categorical feature
        for feature in CATEGORICAL_FEATURES:
            df_test_slice = slice_data(df_test, feature)
            for slice_name, slice_df in df_test_slice.items():
                print(f"Slice: {feature}={slice_name}")
                preds = prediction_pipeline(slice_df, model, encoder, lb, process=True)
                y_true = lb.transform(slice_df[LABEL])
                precision, recall, fbeta = compute_model_metrics(y_true, preds)
                print(f"Precision: {precision}, Recall: {recall}, F1: {fbeta}, Support: {len(slice_df)}")
                print("")
    # Reset printout to console
    sys.stdout = sys.__stdout__