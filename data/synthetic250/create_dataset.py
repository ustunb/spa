import itertools
import random
import pandas as pd
from sra.paths import data_dir

judge_num = 10
item_num = 250
item_ids = list(range(1, item_num+1))

# create a list of all possible pairs of items
pairs = list(itertools.combinations(item_ids, 2))

all_prefs = []
for judge_id in range(1, judge_num+1):
    for (x,y) in pairs:
        pref = random.choice([1, 0, -1])
        all_prefs.append([judge_id, x, y, pref])

df = pd.DataFrame(all_prefs, columns=['judge_id', 'item_id_1', 'item_id_2', 'pref'])

df.to_csv(data_dir / f'synthetic{item_num}'/ f'synthetic{item_num}_pairwise.csv', index=False)

