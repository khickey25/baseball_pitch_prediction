import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
                help="Path to raw csv file to be processed")
ap.add_argument("-o", "--output", required=True,
                help="Path to place processed data")           

args = vars(ap.parse_args())

DATA_INPUT = args['data']
OUTPUT_FOLDER = args['output']


    
def prepare_data(file):
    """Process and clean raw Statcast data into a form suitable for analysis/modeling  
    
    Arguments:
        file {[csv]} -- [Raw data to be processed]
    
    Returns:
        [pandas dataframe] -- [Cleaned dataframe ready to be converted to csv file]
    """    
    #read in the data as a dataframe
    baseball = pd.read_csv(file, index_col = 0)
    
    #due to low amount of missing, simply drop the missing instances.
    baseball = baseball.dropna(how='any') 
    
    #format the target; 0 for ball, 1 for strike
    baseball['description'] = baseball['description'].replace({'blocked_ball': 0, 'ball': 0, "called_strike": 1})
    
    #format the pitch names
    baseball['pitch_name'] = baseball['pitch_name'].replace('Knuckle Curve', 'Curveball')
    
    #baseball['position_x'] = baseball['release_pos_x'] + baseball['pfx_x']
    #baseball['position_z'] = baseball['release_pos_z'] + baseball['pfx_z']

    #filter out the Eephus pitches
    baseball = baseball[baseball.pitch_name != 'Eephus'] 
    
    final_df = baseball.loc[:, ['p_throws','pitch_name', 'release_spin_rate', 
                                'release_pos_x', 'release_pos_y', 'release_pos_z',
                                'vx0', 'vz0', 'vy0', 'sz_top',
                                'sz_bot']]

    #get dummies of the categorical features
    final_df=pd.get_dummies(final_df, prefix = ['handedness', 'pitch_name'])
    
    #add back the target
    final_df['description'] = baseball['description']
    
    #add back the player name for future use when interpreting models
    final_df['player_name'] = baseball['player_name']

    return final_df

def convert_to_csv(dataframe):
    """Convert processed dataframe into two csv files: a training set and a testing set. 
    Will use 85% for the training set, and remaining 15% as a test set. 
    
    Arguments:
        dataframe {[pandas Dataframe]} -- Cleaned pandas dataframe object to be converted into csv file
    """    
    X_train, X_test = train_test_split(dataframe, test_size=.15, random_state=777)
    print('\n Converting dataframes to csv files')
    X_train.to_csv(OUTPUT_FOLDER + 'Statcast_train.csv')
    X_test.to_csv(OUTPUT_FOLDER + 'Statcast_test.csv')
    print('\n Finshed!')


if __name__ == "__main__":
    df = prepare_data(DATA_INPUT)
    convert_to_csv(df)
