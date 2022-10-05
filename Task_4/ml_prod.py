import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

TEST_SIZE = 0.2
RANDOM_STATE = 33
K_FOLDS = 5
SCORER = 'neg_mean_absolute_error'
CV_EXTRACT = 'test_'
NEGATIVE_CONVERTER = -1


def load_data(path: str='Data/sales.csv'):
    """Load the CSV file into a Pandas DataFrame and drop unnecessary column.

    Args:
        path (str): Path to csv file. Defaults to 'Data/sales.csv'.

    Returns:
        pd.DataFrame: Pandas dataframe object
    """
    df = pd.read_csv(path)
    df.drop('Unnamed: 0', axis=1, inplace=True, errors='ingore')
    return df


def load_model(path: str='Task_4/final_model.pkl'):
    """Load the model from a pickle file.

    Args:
        path (str): Path to pickle file containing the model object. Defaults to 'Task_4/final_model.pkl'.

    Returns:
        unpickled: Same type as object stored in file
    """
    return pd.read_pickle(path)


def split_features_and_target(data: pd.DataFrame=None, target: str='estimated_stock_pct'):
    """Splits dataframe into features, X, and target variable, y.

    Args:
        data (pd.DataFrame, optional): Pandas dataframe object. Defaults to None.
        target (str, optional): Target variable to isolate. Defaults to 'estimated_stock_pct'.

    Returns:
        X (pd.DataFrame): Pandas dataframe features
        y (pd.Series): Pandas series target variable
    """
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y


def main(error_score='raise'):
    """Pipeline for loading the data and model, splitting data into training and testing sets, and
    running K-fold cross validation. Then prints the average cross validation score.

    Args:
        data_file (csv): csv file to load
        model_file (pickle): Pickle file containing the model object
        error_score (str, optional): Value to assign to the score if an error occurs in estimator
                                     fitting. Defaults to 'raise'.
    """
    data = load_data()
    X, y = split_features_and_target(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = load_model()

    # Cross validate model fit and predict
    cross_val = cross_val_score(model, X_train, y_train, cv=K_FOLDS, scoring=SCORER, error_score=error_score)

    # Output average model error score
    mae = cross_val.get(CV_EXTRACT+SCORER).mean()*NEGATIVE_CONVERTER
    print(mae)