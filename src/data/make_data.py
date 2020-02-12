#import dependencies
import pybaseball
import pandas as pd
import numpy as np
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
import pathlib
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
                help="path to .txt file containing pitcher names")
ap.add_argument("-o", "--output", required=True,
                help="path to save created raw data csv file")

args = vars(ap.parse_args())   


PITCHER_NAMES = args['file']
DATA_FOLDER = args['output']
#PITCHER_NAMES = pathlib.Path.cwd().parents[1] / 'references' / 'pitcher_names.txt'
#DATA_FOLDER = pathlib.Path.cwd().parents[1] / 'data'


#set up a few constants
#number of pitches
SAMPLE_SIZE = 350

#classes of the target variable
TARGET_CLASSES = ['ball', 'called_strike', 'blocked_ball']

#resulting features we want
FEATURES_TO_KEEP = ['player_name', 'p_throws', 'pitch_name', 'release_speed','release_spin_rate',
                    'release_pos_x', 'release_pos_y',
                    'release_pos_z', 'pfx_x', 'pfx_z', 'vx0','vy0', 'vz0', 
                    'ax', 'ay', 'az', 'sz_top', 'sz_bot', 
                    'release_extension','description']

def read_pitchers(file):
    """Reads in a txt file containing first and last names of pitchers 
    to collect statcast data on. 
    
    Arguments:
        file {.txt file} -- A text file containing the names of pitchers you want to scrape data on.
    
    Returns:
        list -- a list of lists, each element/list containing the first and last name(s) of each pitcher, respectively.
        Note: some players have more than one element to their last name. 
        For instance:
            ["Chris", "Sale"] => ["Chris", "Sale]
            ["Lance", "McCullers", "Jr"] => ["Lance", "McCullers Jr"]
            ["Jorge", "De", "La", "Rosa"] => ["Jorge", "De La Rosa"]
    """    


    print('-' * 10 + 'Reading text file' + '-' * 10 + '\n')
    with open(file) as f:
        names = f.read().split(',')
        for name in names:
            if '\n' in name:
                names = [name.replace('\n', '') for name in names]
            split_names = [name.split(' ') for name in names]

    print('-' *10  + 'Finished reading text file' + '-' * 10 + '\n')
        
    return split_names




def collect_statcast(sample_size, target, features, pitcher_names):
    """Scrapes the Statcast data for each pithcer based on specified criteria; see arguments. 
    
    Arguments:
        sample_size {int} -- the number of pitches to collect for each pithcer
        target {list} -- a list containing the categories desired in the resulting pitch
        features {list} -- a list containing the desired features to keep for the resulting data.
        pitcher_names {list} -- the list of pitcher names from the read_pitchers function.
    
    Returns:
        pandas dataframe -- a pandas dataframe where each row is a single pitch for a particular pitcher
        and each column is a specified feature in the 'features' argument. 
    """
    
    #loop through all the names
    print('Begin scraping \n')
    
    final_data = pd.DataFrame(columns = features)
    
    for i, pitcher in enumerate(pitcher_names):
        if len(pitcher) == 2:
            fname, lname  = pitcher[0], pitcher[1]
        elif len(pitcher) >= 3:
            fname, lname = pitcher[0], " ".join(pitcher[1:])
        else:
            pass
        
        print(f'\n Pitcher Name: {fname} {lname}, #: {i+1}/{len(pitcher_names)}  \n')
        #grap the unique identifier of the pitcher
        player = playerid_lookup(lname, fname)
    
        #to avoid any possible errors, execute following try statement:
        # grab the unique identifier value
        # get all available data in time frame
        # filter data to only have appropriate targets, defined above
        # append particular pitcher to 'master' dataframe
        #if any of these steps fail, particularly the grabbing of 'ID'
        #pass on to next pitcher
        try:
            ID = player['key_mlbam'].iloc[player['key_mlbam'].argmax()]
            df = statcast_pitcher('2018-03-29', '2018-09-30', player_id = ID)
            df = df[df['description'].isin(target)].sample(sample_size, random_state=2019)
            final_data = final_data.append(df[features], ignore_index = True)
            
        except ValueError:
            pass


    print('Finsihed Scraping')
    return final_data
    


def convert_to_csv(dataframe):
    """Converts the resulting pandas dataframe created in the collect_statcast function 
    into a csv file in the data/raw directory. 
    
    Arguments:
        data {pandas Dataframe} -- the dataframe to be converted. 
    """    
    print('\n Converting dataframe to csv file')
    dataframe.to_csv(str(DATA_FOLDER) + 'Statcast_data.csv')
    print('\n Finshed!')


def main():
    
    names = read_pitchers(PITCHER_NAMES)
    pitchers = collect_statcast(SAMPLE_SIZE, TARGET_CLASSES, FEATURES_TO_KEEP, names)
    convert_to_csv(pitchers)


if __name__ == "__main__":
    main()
               