def prepare_data(file):
    '''give the path to the Statcast data file
    do processing steps'''
    
    import pandas as pd
    import numpy as np
    
    #read in the data as a dataframe
    baseball = pd.read_csv(file, index_col = 0)
    
    #format the target
    baseball['description'] = baseball['description'].replace({'blocked_ball': 0, 'ball': 0, "called_strike": 1})
    
    #
    #baseball['position_x'] = baseball['release_pos_x'] + baseball['pfx_x']
    #baseball['position_z'] = baseball['release_pos_z'] + baseball['pfx_z']
    
    #format the pitch names
    baseball['pitch_name'] = baseball['pitch_name'].replace('Knuckle Curve', 'Curveball')
    
    #filter out the Eephus pitches
    baseball = baseball[baseball.pitch_name != 'Eephus'] 
    
    final_df = baseball.loc[:, ['p_throws','pitch_name', 'release_spin_rate', 
                                'release_pos_x', 'release_pos_y', 'release_pos_z',
                                'vx0', 'vz0', 'vy0', 'sz_top',
                                'sz_bot', 'description']]

    #get dummies of the categorical features
    final_df=pd.get_dummies(final_df, prefix = 'pitch')
    
    #add back the player name for future use when interpreting models
    final_df['player_name'] = baseball['player_name']
    
    #due to low amount of missing, simply drop the missing instances.
    final_df = final_df.dropna()
    return final_df
