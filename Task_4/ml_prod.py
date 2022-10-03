# OUTLINE

# 1. Import statements
# 2. Constants, if needed
# 3. Define functions for modeling
# 4. "main" function that runs everything


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

# train test split
# fit on training
# predict testing
# functions for returning the cross val score, prediction


def load_data(file):
    """Load data from CSV to DataFrame.

    Args:
        file (csv): File to be of type csv.
    """
    return pd.read_csv(file)


def model_fitting(df: pd.DataFrame=None):
    """_summary_

    Args:
        df (pd.DataFrame, optional): _description_. Defaults to None.
    """
    X_train, X_test, y_train, y_test = train_test_split(df, test_size=0.2, random_state=433)
    pass