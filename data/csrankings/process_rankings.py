#starting from scratch for processing rankings, since the other file didn't work...

"""
This file is used to create CSV files for csrankings
must be run from this directory
"""

import pandas as pd
import numpy as np

data_name = 'csrankings'
# values assigned to different kinds of preferences
PREFERENCE_TYPES = {
    'prefer_1': 1, # prefers item_1 to item_2
    'prefer_2': -1, # prefers item_2 to item_1
    'prefer_none': 0, # equivalent between item_1 and item_2
    }

#access files
def file_name(k):
    return '{}_{}.csv'.format(data_name, k)

def ranks_to_prefs(ranks, items):
    """
    split a table of ranks into a table of pairwise preferences
    :param ranks: pd.DataFrame with [judge_id, item_id, rank]
    :param items: full set of item_ids (since some items might be unranked by some judges)
    :return: prefs: pd.DataFrame with [judge_id, item_id_1, item_id_2, pref]
    """
    df = pd.DataFrame(ranks)

    judges = df['judge_id'].unique()

    df.set_index('judge_id', inplace = True)
    df = df.join(df, how = "inner", lsuffix = "_1", rsuffix = "_2")
    keep_idx = df['item_id_1'] < df['item_id_2']
    df = df[keep_idx]

    df['pref'] = float('nan')
    df.loc[df['rank_1'] < df['rank_2'], 'pref'] = PREFERENCE_TYPES['prefer_1']
    df.loc[df['rank_2'] < df['rank_1'], 'pref'] = PREFERENCE_TYPES['prefer_2']
    df.loc[df['rank_2'] == df['rank_1'], 'pref'] = PREFERENCE_TYPES['prefer_none'] #need to re-examine this here; it doesn't match pareto optimality
    assert ~df['pref'].isna().any()

    df['pref'] = df['pref'].astype(int)
    df.drop(['rank_1', 'rank_2'], axis = 1, inplace = True)
    df.reset_index(inplace = True)
    return df


ranks = pd.read_csv(file_name('ranks'))
items = pd.read_csv(file_name('items'))['item_id'].values
prefs = ranks_to_prefs(ranks, items)
prefs.to_csv(file_name('prefs'), index = False)
assert all(prefs == pd.read_csv(file_name('prefs')))