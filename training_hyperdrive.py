from sklearn.ensemble import GradientBoostingRegressor
import math
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def clean_data(original_df):
    
    # Get rid of cols with missing values.
    cols_with_missing = [col for col in original_df.columns if original_df[col].isnull().any()] 
    object_cols = [col for col in original_df.columns if original_df[col].dtype == "object"]

    # Split data/predictors from label and return both.
    X = original_df.drop(['SalePrice'] + object_cols + cols_with_missing, axis=1)


    y = original_df[['SalePrice']]
    return (X, y)



def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_estimators', type=int, default=100)
    
    args = parser.parse_args()

    run = Run.get_context()
    run.log("Learning Rate:", np.float(args.learning_rate))
    run.log("N Estimators:", np.int(args.n_estimators))
    
    # Get the workspace.
    ws = run.experiment.workspace

    # get data
    key = 'AmesHousingData'
    if key in ws.datasets.keys():
        data = ws.datasets[key]
        print(f"Dataset {key} located and loaded in.")
    else:
        print(f"Cannot find {key}.")

    # Load to pandas original_df
    original_df = data.to_pandas_dataframe()
    
    
    sale_price_mean = original_df['SalePrice'].mean()
    
    
    X, y = clean_data(original_df)


    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    model = GradientBoostingRegressor(n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=1, random_state=0, loss='huber').fit(X_train, y_train)


    run.log("normalized_root_mean_squared_error", np.float(model.score(X_test, y_test)))

    
    if "output" not in os.listdir():
        os.mkdir("./output")


    # run.log("normalized root mean squared error", model.score(x_test, y_test))

    # Save the model into run history
    joblib.dump(model, 'output/model.joblib')


if __name__ == '__main__':
    main()