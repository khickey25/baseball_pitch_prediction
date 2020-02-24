import numpy as np 
import pandas as pd 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
import matplotlib.pyplot as plt
import joblib
import argparse
from joblib import dump, load
import pathlib
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


#create command line parser
#taking three arguments: specific classifier to tune, the path to the dataset
#and a path to save the tuned models
ap = argparse.ArgumentParser(description="Tune Hyperparameters of specific classifiers to generate best performing models")

ap.add_argument('-c', '--classifier', required=True,
                help='Path to intial, untuned classifier in models directory')

ap.add_argument('-d', '--data', required=True,
                help='Path to processed training dataset to tune hyperparameters')

ap.add_argument('-o', '--output',
                help='Path to saved final tuned model')

args = vars(ap.parse_args())

CLF = args['classifier']
#grab the name of the model from the model path
#ex: models/saved_models/Logistic_Regression.joblib => 'Logistic_Regression'
CLF_NAME = str(CLF).split('.')[0].split('/')[-1].lower()

#designate the target and predictor variables
TARGET = 'description'
PREDICTORS = ['release_spin_rate', 'release_pos_x', 'release_pos_y',
       'release_pos_z', 'vx0', 'vz0', 'vy0', 'sz_top', 'sz_bot',
       'handedness_L', 'handedness_R', 'pitch_name_2-Seam Fastball',
       'pitch_name_4-Seam Fastball', 'pitch_name_Changeup',
       'pitch_name_Curveball', 'pitch_name_Cutter', 'pitch_name_Sinker',
       'pitch_name_Slider', 'pitch_name_Split Finger']

#create a master dictionary, containing the specific parameters for each model to tune on
PARAM_GRIDS = {'lr': dict(model__C=np.logspace(-4, 4, 15)),
                'dt': {
        'model__criterion': Categorical(['gini', 'entropy']),
        'model__max_depth': Integer(1,4),
        #'model__max_features': Categorical('auto', 'sqrt', 'log2')
                }
                
                } 

#main evaluation metric, also designating a refit_metric
#for compatibality in returning the best parameter. 
SCORES = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'f1':'f1'}
REFIT_SCORER = 'AUC'

ITERATIONS = 5


def tune_model(model, dataset):
    
    
    #load in serialized model specified in command line argument. 
    print('\n Loading Classifier \n')
    clf = joblib.load(model)
    
    #load in csv file specified in command line argument. 
    print('Loading in Dataset \n')
    df = pd.read_csv(args['data'])

    
    #find which model name was specified and tune said model. 
    if CLF_NAME == 'logistic_regression':
        #perform hyperparameter serch
        grid_clf = GridSearchCV(clf, param_grid=PARAM_GRIDS['lr'], scoring=SCORES, refit=REFIT_SCORER, 
                                verbose=0, n_jobs=-1)
        
        print('Performing Grid Search \n')
        grid_clf.fit(df[PREDICTORS], df[TARGET])

        result_cols = ['params', 'mean_test_AUC', 'std_test_AUC', 'rank_test_AUC',
                    'mean_test_Accuracy', 'std_test_Accuracy', 'rank_test_Accuracy',
                    'mean_test_f1', 'std_test_f1', 'rank_test_f1']
        
        #grab the results as a pandas dataframe for easier viewing
        results = pd.DataFrame(grid_clf.cv_results_).loc[:, result_cols].sort_values('rank_test_' + REFIT_SCORER)
        print('Printing overall results \n')
        print(results)

        #grab the best performing model object and return it. 
        print('Printing best performing classifier \n')
        print(grid_clf.best_estimator_)
        return grid_clf.best_estimator_

    elif CLF_NAME == 'decision_tree':
        
        #create a bayesian optimized hyperparameter search
        bayes_cv_tuner_dt = BayesSearchCV(estimator = clf,
        search_spaces = PARAM_GRIDS['dt'],    
            scoring = 'roc_auc',
            cv = 5,
            n_jobs = -1,
            n_iter = ITERATIONS,   
            verbose = 1,
            refit = True,
            random_state = 777,
            return_train_score=True
                                    )
        #fit the hyperparameter search
        result = bayes_cv_tuner_dt.fit(df[PREDICTORS], df[TARGET])
        
        #save results to a dataframe
        results_df = pd.DataFrame(result.cv_results_).sort_values('rank_test_score')
        print(results_df)

        #return the best estimator
        return result.best_estimator_

def save_tuned_model(clf):
     
    #create repository if path specied in command line argument does not yet exist
    if args['output']:
        if not pathlib.Path(args['output']).exists():
            pathlib.Path(args['output']).mkdir(exist_ok=True, parents=True)
    
    #save the serialized tuned model via joblib
    print(f"Saving model to: {args['output']} \n")
    if CLF_NAME == 'logistic_regression':
        dump(clf, str(args['output']) + 'Tuned_Logistic_Regression.joblib')

    elif CLF_NAME == 'decision_tree':
        dump(clf, str(args['output']) + 'Tuned_Decision_Tree.joblib')


if __name__ == "__main__":
    tuned_model = tune_model(model=args['classifier'], dataset=args['data'])
    save_tuned_model(tuned_model)
