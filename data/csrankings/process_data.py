import pandas as pd
import numpy as np

### Must be run from the csrankings directory

############ make judges dataframe ################
AI = 0#'AI-just-artificial-intelligence'
VISION = 1#'AI-Vision'
ML = 2#'AI-ML-data-mining'
NLP = 3
WEB = 4

judge_set = [AI, VISION, ML, NLP, WEB]

JUDGE_MAP = {
    AI: 'AI-just-artificial-intelligence',
    VISION: 'AI-Vision',
    ML: 'AI-ML-data-mining',
    NLP: 'AI-NLP',
    WEB: 'AI-Web-Retrieval'
}

judges = pd.DataFrame({'judge_id': pd.Series(dtype=int), "judge_name": [], 'judge_affiliation': []})

for judge in judge_set: 
    judges = judges.append({'judge_id': judge,'judge_name': JUDGE_MAP[judge],  'judge_affiliation': str("none")}, ignore_index=True)

judges.to_csv('cs_rankings_judges.csv', index=False)

######### next make items: ######
df = pd.DataFrame(pd.read_excel('csrankings demo.xlsx', 'AI-past-decade-9-13-22-CSR')) #load the aggregate sheet

all_schools_orig = df['Institution'] #get the set of schools using the raw xl formatting

#process a bit to remove unnecessary part of the string
all_schools = all_schools_orig.copy()
for i in range(len(all_schools)): 
    all_schools[i] = all_schools[i].split("  closed")[0].split("► ")[1]
items = pd.DataFrame({'item_id': np.arange(len(all_schools)), 'item_name': all_schools})

#save csv
items.to_csv('cs_rankings_items.csv', index=False)

# go from school name to item index, based on player_array
def lookup_item(item_name, item_array = all_schools_orig): 
    return np.where(item_array == item_name)[0][0]

######## make rankings (existing script can then make prefs from that) ####

rankings = pd.DataFrame(columns=["judge_id", 'item_id', 'rank'])

for judge in judge_set: 
    judge_prefs = pd.DataFrame(pd.read_excel('csrankings demo.xlsx', JUDGE_MAP[judge])) #load the aggregate sheet
    for idx, row in judge_prefs.iterrows(): 
        judge_id = judge
        item_id = lookup_item(row['Institution'] , all_schools_orig)
        rank = row['#']
        to_add = pd.DataFrame({"judge_id": [judge_id], "item_id": [item_id], "rank": [rank]})
        rankings = pd.concat([rankings, to_add])

rankings.to_csv('cs_rankings_ranks.csv', index=False)