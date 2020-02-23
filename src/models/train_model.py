import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imb_Pipe
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import pathlib
import argparse
from joblib import dump, load

ap = argparse.ArgumentParser(description="Train, evaluate, and store classification models")

ap.add_argument('-i', '--input', required=True, 
                help='Input path to train data csv file')

ap.add_argument("-m", "--model", required=True,
                help="Specific model to train and determine results. Accepted values are: lr, dt, rf, xg")

ap.add_argument("-o", "--output",
                help='Path to save models in pickle format for future use')

args = vars(ap.parse_args())

DATA = args['input']
PATH_TO_SAVE_MODELS = pathlib.Path(args['output'])

def train_model(csv):
    """Trains a given classifier on the Statcast_train.csv dataset and returns 5-fold cross validation results. 
    Saves the model to specified destination via joblib.
    
    Arguments:
        csv {[csv file]} -- Designated training dataset to train classifiers.
    
    Returns:
        sklearn pipeline -- Pipeline containing two elements: a random undersampler to balance the target via the imblearn package,
        and one of four designated scikit-learn classifier objects.
    """    
    
    df = pd.read_csv(csv)

    print(f"\n Dataframe shape: {df.shape[0]} rows x {df.shape[1]} columns \n")

    #filter out player_name and target
    print("Dropping unnecessary features \n")
    X = df.drop(columns = ['player_name', 'description'])
    y = df['description']

    #build the pipeline objects
    lg = LogisticRegression(random_state=777)
    dt = DecisionTreeClassifier(max_depth=4, random_state=777)
    rf = RandomForestClassifier(random_state=777)
    xg = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, verbosity=0, random_state=777)
    sampler = RandomUnderSampler(sampling_strategy='auto', random_state=777)
    
    if args['model'] == 'lr':   
        pipe = imb_Pipe([('sampler', sampler), ('model', lg)])
    elif args['model'] == 'dt':
        pipe = imb_Pipe([('sampler', sampler), ('model', dt)])
    elif args['model'] == 'rf':
        pipe = imb_Pipe([('sampler', sampler), ('model', rf)])
    elif args['model'] == 'xg':
        pipe = imb_Pipe([('sampler', sampler), ('model', xg)])

    print(f"Training {pipe['model']} \n" + "*" *50 + '\n')
    results = cross_validate(pipe, X, y, 
                                scoring = ['accuracy', 'f1', 'roc_auc'], 
                                cv=5, return_estimator=True, return_train_score=True)
    print("Finished 5-fold Cross Validation. Printing Results Below \n")

    for result in ['train_accuracy', 'test_accuracy', 'train_f1', 'test_f1', 'train_roc_auc', 'test_roc_auc']:
        print(f"Average {result}: {results[result].mean():.3f} +/- {results[result].std() *2:.3f} ")
        print(f"{result} scores: {results[result]} \n")
    
    
    return pipe

def save_model(clf):
    """Saves the newly formed pipeline object into a joblib file to store the pipeline object for further analysis.
    
    Arguments:
        clf {[sklearn pipeline]} -- scikit-learn pipeline object trained on the training data. 
        Will be saved to path designated in file arguments. 
    """    
    print("Saving Model")
    if not PATH_TO_SAVE_MODELS.exists():
            PATH_TO_SAVE_MODELS.mkdir(exist_ok=True, parents=True)
    
    if args['model'] == 'lr':
        dump(clf, str(PATH_TO_SAVE_MODELS) + "/Logistic_Regression.joblib")  
        
    elif args['model'] == 'dt':
        dump(clf, str(PATH_TO_SAVE_MODELS) + "/Decision_Tree.joblib")

    elif args['model'] == 'rf':
        dump(clf, str(PATH_TO_SAVE_MODELS) + "/Random_Forest.joblib")

    elif args['model'] == 'xg':
        dump(clf, str(PATH_TO_SAVE_MODELS) + "/XGB.joblib")
    


if __name__ == "__main__":
    classifier = train_model(DATA)
    save_model(classifier)
    print(f"\n" + "*" *20 + " Completed " + "*" *20 + "\n")
