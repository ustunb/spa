from dataclasses import dataclass, field, InitVar
from itertools import combinations
import pandas as pd
import numpy as np
from spa.utils import to_preference_value
from spa.data import PreferenceDataset

@dataclass
class Ranking:
    df: InitVar[pd.DataFrame]
    items: set = field(init = False)
    ranks: pd.DataFrame = field(init = False, default_factory = pd.DataFrame)
    pref_df: pd.DataFrame = field(init = False, default_factory = pd.DataFrame)
    n_tiers: int = field(init = False)
    n_pairs: int = field(init=False)
    n_comparisons: int = field(init=False)
    n_abstentions: int = field(init = False)

    def __post_init__(self, df: pd.DataFrame):
        assert df.shape[0] >= 1, "df must include at least 1 item"
        assert 'item_name' in df.columns, "df must include 'item_name'"
        assert 'rank' in df.columns, "df must include 'rank'"

        self.ranks = df[['item_id', 'item_name', 'rank']]
        self.items = set(df['item_name'].unique())
        n_items = len(self.items)
        self.item_pairs = list(combinations(self.items, 2))
        self.n_pairs = len(self.item_pairs)

        # Check ranks
        assert df['rank'].dtype.kind in 'if', "Ranks must be numeric"
        df['rank'] = df['rank'].astype(int)  # Convert ranks to integers

        ranks = list(set(df['rank']))
        assert np.min(ranks) >= 1, "Ranks must start from 1"
        assert np.max(ranks) <= n_items, "Ranks must not exceed the number of items"
        # assert (np.array(ranks) == np.arange(np.min(ranks), np.max(ranks) + 1)).all(), "Ranks do not match expected range"
        self.n_tiers = np.max(ranks)

        # Count comparisons
        items_per_rank = df['rank'].value_counts(sort=False)
        within_rank_comparisons = items_per_rank.apply(lambda m: m * (m - 1) / 2)
        self.n_comparisons = self.n_pairs - within_rank_comparisons.sum()
        self.n_abstentions = within_rank_comparisons.sum()

        pairwise_data = []
        # print(df)
        for idx, row in df.iterrows():
            for idx2, row2 in df.iterrows():
                if idx < idx2:
                    item_id_1, item_id_2 = row['item_id'], row2['item_id']
                    pref_value = to_preference_value(row['rank'], row2['rank'])
                    if item_id_1 > item_id_2:
                        item_id_1, item_id_2 = item_id_2, item_id_1
                        pref_value *= -1

                    pairwise_data.append({'item_id_1': item_id_1, 'item_id_2': item_id_2, 'pref': pref_value})


        self.pref_df = pd.DataFrame(pairwise_data)
        #sort on item_id_1 and item_id_2 where item_id_1 < item_id_2
        self.pref_df = self.pref_df.sort_values(by=['item_id_1', 'item_id_2'])


        assert self.__check_rep__()

    def __check_rep__(self):
        pref_values = self.pref_df['pref'].unique()
        assert all(p in [-1, 0, 1] for p in pref_values), "All preferences should be -1, 0, or 1"
        assert self.n_tiers <= len(self.items), "Number of tiers should not exceed the number of items"
        return True

    def is_compatible_with(self, other):
        """Return True if this ranking preferences are compatible with another Rankings preferences"""
        assert self.is_comparable_to(other), "Rankings are not comparable"

        failures = []

        for _, row in self.pref_df.iterrows():
            x, y = row['item_id_1'], row['item_id_2']
            pref = row['pref']

            if pref == 1:  # R ranks x > y
                if (x, y) in other.pref_df.values:
                    other_pref = other.pref_df[(other.pref_df['item_id_1'] == x) & (other.pref_df['item_id_2'] == y)]['pref'].values[0]
                    if other_pref != 1:  # S ranks x = y or x < y
                        failures.append((x, y))
                elif (y, x) in other.pref_df.values:
                    other_pref = other.pref_df[(other.pref_df['item_id_1'] == y) & (other.pref_df['item_id_2'] == x)]['pref'].values[0]
                    if other_pref != -1:  # S ranks y = x or y < x
                        failures.append((y, x))
                else:
                    raise ValueError(f"Pair ({x}, {y}) not found in S")

        return len(failures) == 0

    def is_comparable_to(self, other):
        """Return True if we can compare this ranking to another ranking"""
        out = isinstance(other, Ranking) and self.items == other.items
        return out

    def __eq__(self, other):
        out = (
            self.items == other.items and
            self.ranks.equals(other.ranks)
        )
        return out

    def get_prefs_from_df(self, df = None):

        if df is None:
            df = self.ranks

        prefs = {}
        for idx, row in df.iterrows():
            for idx2, row2 in df.iterrows():
                if idx < idx2:
                    prefs[(row['item_id'], row2['item_id'])] = np.sign(row['rank'] - row2['rank'])
                    prefs[(row2['item_id'], row['item_id'])] = -np.sign(row['rank'] - row2['rank'])

        return prefs

    def __len__(self):
        return self.n_comparisons

    def __repr__(self):
        return f'Ranking<{len(self.items)} items, {self.n_tiers} tiers>'

    def __str__(self):
        self.ranks = self.ranks.sort_values(by='rank')
        return self.ranks['item_name'].to_string()

    def __getitem__(self, key):
        out = None
        if isinstance(key, tuple):
            out = self.get_pref(key[0], key[1])
        elif isinstance(key, int):
            out = self.get_rank(key)
        return out

    def get_pref(self, item_id, other_id):
        """
        """
        isinstance(item_id, int) and item_id > 0
        isinstance(other_id, int) and other_id > 0

        df = self.ranks

        #get 'rank' of item_id and other_id from self.ranks
        rank_item_item = df.loc[df['item_id'] == item_id, 'rank'].values[0]
        rank_other_item = df.loc[df['item_id'] == other_id, 'rank'].values[0]

        return 1 if rank_item_item > rank_other_item else -1 if rank_item_item < rank_other_item else 0


    def get_rank(self, item_id):
        isinstance(item_id, int) and item_id > 0
        out = 1 #value between 1 to n_iters
        return out


    def user_stats(self, data):
        """Generates statistics for each judge including preferences, overruled cases, and agreements."""
        assert isinstance(data, PreferenceDataset), "Data must be a PreferenceDataset"

        # Initialize lists to store statistics for each judge
        judge_ids = []
        n_preferences = []
        n_overruled = []
        n_agreement = []

        # Iterate over each judge's preferences\
        judge_names = data.judges['judge_name'].unique()

        for judge_id in data.judges['judge_id']:
            judge_prefs = data.prefs[data.prefs['judge_id'] == judge_id]

            # Count the number of preferences, overruled cases, and agreements
            n_pref = len(judge_prefs)
            overruled = 0
            agreement = 0
            for _, row in judge_prefs.iterrows():
                pair = (row['item_id_1'], row['item_id_2'])
                pair_pref = self.get_pref(pair[0], pair[1])

                if row['pref'] == pair_pref:
                    agreement += 1
                elif row['pref'] != 0:
                    overruled += 1

                #check here to see if a flip is needed

            # Append statistics to the lists
            judge_ids.append(judge_id)
            n_preferences.append(n_pref)
            n_overruled.append(overruled)
            n_agreement.append(agreement)

        # Create a DataFrame from the lists
        df_stats = pd.DataFrame({
            'judge_name': judge_names,
            'n_preferences': n_preferences,
            'n_overruled': n_overruled,
            'n_agreement': n_agreement
        })

        return df_stats
    def get_prefs(self):
        for i, row in self.pref_df.iterrows():
            if row['item_id_1'] > row['item_id_2']:
                self.pref_df.at[i, 'item_id_1'], self.pref_df.at[i, 'item_id_2'] = row['item_id_2'], row[
                    'item_id_1']
                self.pref_df.at[i, 'pref'] *= -1

        return self.pref_df

    def overall_stats(self, data):
        n_agreements = 0
        n_disagreements = 0
        median_change = 0
        min_change = 0
        max_change = 0

        for _, row in data.prefs.iterrows():
            if np.isnan(row['pref']):
                continue

            item_pair = (row['item_id_1'], row['item_id_2'])

            ranking_pref = self.get_pref(item_pair[0], item_pair[1])

            if row['pref'] == ranking_pref:
                n_agreements += 1
            else:
                n_disagreements += 1

        return n_agreements, n_disagreements

    def item_stats(self, data):
        """Generates statistics for each item including preferences, overruled cases, and agreements."""
        assert isinstance(data, PreferenceDataset), "Data must be a PreferenceDataset"

        # Initialize lists to store statistics for each item
        item_stats = []

        # Iterate over each item
        for item_id in data.items['item_id']:
            item_prefs = data.prefs[(data.prefs['item_id_1'] == item_id) | (data.prefs['item_id_2'] == item_id)]

            # Count the number of preferences, overruled cases, and agreements
            n_pref = len(item_prefs)
            disagreements = 0
            agreements = 0
            comparisions = 0
            abstentions = 0

            for _, row in item_prefs.iterrows():
                if np.isnan(row['pref']):
                    abstentions += 1
                    continue

                comparisions += 1

                pair = (row['item_id_1'], row['item_id_2'])
                pair_pref = self.get_pref(pair[0], pair[1])

                if row['pref'] == pair_pref:
                    agreements += 1
                elif row['pref'] != 0:
                    disagreements += 1

                #check here to see if a flip is needed

            # Append statistics to the list
            item_stats.append({
                'dissent_rate': disagreements / n_pref,
                'item_id': item_id,
                'item_name': data.items[data.items['item_id'] == item_id]['item_name'].values[0],
                'rank': self.get_rank(item_id),
                'n_comparisons': comparisions,
                'n_abstentions': abstentions,
                'n_disagreements': disagreements,
                'n_agreements': agreements
            })

        # Create a DataFrame from the list
        df_stats = pd.DataFrame(item_stats)

        return df_stats


