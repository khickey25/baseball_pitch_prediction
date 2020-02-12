#start with all dependencies
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import Pipeline as imb_Pipe
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pathlib
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('-d', '--data', required=True, 
                help='Path to processed data')

args = vars(ap.parse_args())

DATA = args['data']

def train_logistic_regression(csv):
    
    df = pd.read_csv(csv)

    print(f"Dataframe shape: {df.shape[0]} rows x {df.shape[1]} columns \n")

    #filter out player_name and target
    print("Dropping unnecessary features \n")
    X = df.drop(columns = ['player_name', 'description'])
    y = df['description']

    print("Creating data partitions \n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 777)

    #build the pipeline object
    logit_reg = LogisticRegression()
    sampler = RandomUnderSampler(sampling_strategy='auto', random_state=777)

    logit_pipe = imb_make_pipeline(sampler, logit_reg)

    print(f"*" *20 + "  Training Logistic Regression " + "*" *20 + '\n')
    logit_pipe_results = cross_validate(logit_pipe, X_train, y_train, 
                                scoring = ['accuracy', 'f1', 'roc_auc'], 
                                cv=5, return_estimator=True, return_train_score=True)
    print("Finished 5-fold Cross Validation. Printing Results Below \n")

    for result in ['train_accuracy', 'test_accuracy', 'train_f1', 'test_f1', 'train_roc_auc', 'test_roc_auc']:
        print(f"Mean {result} Value: {np.mean(logit_pipe_results[result])}")
        print(f"{result} scores: {logit_pipe_results[result]} \n")
    print(f"\n" + "*" *20 + " Completed " + "*" *20)

if __name__ == "__main__":
    train_logistic_regression(DATA)