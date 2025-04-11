from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
from copy import copy
import dill
from dataclasses import dataclass, field
from spa.utils import check_names, to_preference_value
import re
from typing import Optional
pd.options.mode.chained_assignment = None

# IO constants
TABLE_TYPES = ('items', 'judges', 'df')
PREFERENCE_TYPES = ('pairwise', 'rankings', 'ratings')

# Values assigned to different kinds of preferences
PREFERENCE_VALUES = {
    'prefer_1': 1,  # prefers item_1 to item_2
    'prefer_2': -1,  # prefers item_2 to item_1
    'prefer_none': 0,  # equivalent between item_1 and item_2
    'prefer_missing': np.nan,  # missing preference
    }

# Parser
def parse_preference_type(text):
    match = re.search(r'_(?!.*_)', text)
    if match:
        return text[match.end():]
    return None


@dataclass
class PreferenceDataset:
    """Class to represent a dataset of pairwise preferences"""
    prefs: pd.DataFrame
    items: pd.DataFrame
    judges: pd.DataFrame
    rng: np.random.Generator = field(init=False)
    prefs_dict: dict
    judge_prefs: dict
    preference_type: str
    samples_indices: list = field(default_factory=list)
    seed: Optional[int] = 2338

    def __post_init__(self):
        self.items = pd.DataFrame(self.items)
        self.judges = pd.DataFrame(self.judges)
        self.rng = np.random.default_rng(self.seed)

        self.__check_rep__()

    @property
    def n_items(self):
        """Number of items"""
        return len(self.items)

    @property
    def n_judges(self):
        """Number of judges"""
        return len(self.judges)

    @property
    def n_pairs(self):
        """Number of pairwise comparisons"""
        return self.n_items * (self.n_items - 1) // 2

    @property
    def n_prefs(self):
        """Number of pairwise preferences from all judges"""
        return len(self.prefs)

    @property
    def missingness_per_judge(self):
        """Number of judges with missing preferences"""
        judge_prefs = self.prefs.groupby(['judge_id']).size().reset_index(name='count')
        out = self.n_pairs - judge_prefs['count']
        return out

    @property
    def missingness_per_item(self):
        """Number of items with missing preferences"""
        item_prefs = self.prefs.groupby(
                ['item_id_1', 'item_id_2']).size().reset_index(name = 'count')
        out = len(self.judges['judge_id'].unique()) - item_prefs['count']
        return out

    @property
    def n_samples(self):
        """Number of samples"""
        return len(self.samples_indices)

    @property
    def id_to_name(self):
        """Number of items"""
        #create a dictionary of item_id to item_name
        item_dict = dict(zip(self.items['item_id'], self.items['item_name']))
        return item_dict

    def filter_to(self, judge_ids):
        """
        :param judge_ids: List of judge IDs to filter the dataset to
        :return: new dataset with preferences for only one judge
        """
        judge_idx = self.prefs['judge_id'].isin(judge_ids)
        out = PreferenceDataset(prefs=self.prefs[judge_idx], items=self.items, judges=self.judges[self.prefs[judge_idx]])
        return out

    def __eq__(self, other):
        """
        Checks if all properties and attributes are the same
        :param other: another PreferenceDataset object
        :return: True if all attributes are the same, False otherwise
        """
        out = ( isinstance(other, PreferenceDataset) and
                self.items.equals(other.items) and
                self.judges.equals(other.judges) and
                self.prefs.equals(other.prefs)
        )
        return out

    def __check_rep__(self):
        """Check representation invariants"""
        # Check item table
        assert 'item_id' in self.items, "items should contain 'item_id' column"
        assert self.items['item_id'].ge(0).all(), "item_ids should be non-negative"
        assert self.items['item_id'].apply(lambda x: isinstance(x, int)).all(), "item_ids should be integers"
        assert self.items['item_id'].is_unique, "item_ids are not unique"

        # Check judge table
        assert 'judge_id' in self.judges, "judges missing 'judge_id' column"
        assert self.judges['judge_id'].ge(0).all(), "judge_ids should be non-negative"
        assert self.judges['judge_id'].apply(lambda x: isinstance(x, int)).all(), "judge_ids should be integers"
        assert self.judges['judge_id'].is_unique, "judge_ids are not unique"
        # assert np.all(self.judges.judge_id == np.unique(self.df.judge_id)), "judge_ids in df != judge_ids in judges"

        # Check preference table
        for column in ['judge_id', 'item_id_1', 'item_id_2', 'pref']:
            assert column in self.prefs, f"df missing '{column}' column"
        assert all(self.prefs['item_id_1'] != self.prefs['item_id_2']), "Preferences should be between distinct items"
        assert self.prefs['pref'].isin(PREFERENCE_VALUES.values()).all(), "Invalid preference values"
        assert self.prefs.groupby(['judge_id', 'item_id_1', 'item_id_2']).size().le(1).all(), "More than one preference per directional pair"

        # Check that item_ids match in df
        item_ids = np.unique(np.concatenate([self.prefs['item_id_1'].unique(), self.prefs['item_id_2'].unique()])).astype(int)
        assert np.array_equal(np.sort(self.items['item_id']), np.sort(item_ids)), 'item_ids in items != item_ids in df'

        tuple_sample_indices = [tuple(sample_indices) for sample_indices in self.samples_indices]
        assert len(set(tuple_sample_indices)) == len(tuple_sample_indices), "Sample indices are not unique"

        #check that each len(sample index) is within 1 of len(data.perfs)* 0.9
        for sample_indices in self.samples_indices:
            assert abs(len(sample_indices) - int(0.9 * len(self.prefs))) <= 1, "Sample indices are not 90% of the data"

        return True

    #### Sampling ####
    import numpy as np

    def _generate_single_sample_indices(self, sample_idx: int, frac: float = 0.9):
        """
        Generates sample indices for a single sample index on-demand.

        :param sample_idx: The index of the sample for which to generate indices.
        :param frac: Fraction of the dataset to sample.
        :return: A list of indices for the specified sample.
        """
        assert 0.0 <= frac <= 1.0, "Fraction must be between 0 and 1"
        n_to_sample = min(int(frac * len(self.prefs)), len(self.prefs))

        rng = np.random.default_rng(self.seed + sample_idx)  # Seed based on sample_idx
        all_indices = np.arange(len(self.prefs))
        sampled_indices = rng.choice(all_indices, size=n_to_sample, replace=False).tolist()
        return sampled_indices

    def retrieve_single_sample(self, sample_idx: int, frac: float = 0.9):
        """
        Retrieves a specific sample by generating its indices on-demand.

        :param sample_idx: Index of the sample to retrieve.
        :param frac: Fraction of the dataset to sample.
        :return: A DataFrame with the sampled preferences.
        """
        sample_indices = self._generate_single_sample_indices(sample_idx, frac)
        sampled_prefs = self.prefs.iloc[sample_indices]
        return sampled_prefs

    def generate_sample_indices(self, num_samples=100, frac=0.9, seed=2338):
        """
        Generate and store sample indices for reproducibility.
        :param num_samples: Number of samples to generate.
        :param frac: Fraction of the dataset to sample in each instance.
        :param seed: Seed for random number generator (optional).
        """
        assert 0.0 <= frac <= 1.0, "Fraction must be between 0 and 1"
        n_to_sample_any = min(int(frac * len(self.prefs)), len(self.prefs))

        # Use a single vectorized sampling instead of looping
        all_indices = np.arange(len(self.prefs))

        # Generate num_samples number of samples of length n_to_sample_any
        sampled_indices = [self.rng.choice(all_indices, size=n_to_sample_any, replace=False).tolist() for _ in
                           range(num_samples)]

        # Store only the indices (as lists of integers)
        self.samples_indices = sampled_indices

    def retrieve_sample(self, sample_idx: int):
        """
        Retrieve a specific sample based on stored indices.
        :param sample_idx: Index of the sample to retrieve.
        :return: A new PreferenceDataset object with the sampled preferences.
        """
        if not (0 <= sample_idx < len(self.samples_indices)):
            raise IndexError("Sample index out of range.")

        sample_indices = self.samples_indices[sample_idx]
        sampled_prefs = self.prefs.iloc[sample_indices]

        return sampled_prefs

    def retrieve_gaming_sample(self, sample_idx: int, prefs = None):
        """
        Retrieve a specific sample based on stored indices.
        :param sample_idx: Index of the sample to retrieve.
        :return: A new PreferenceDataset object with the sampled preferences.
        """

        if not (0 <= sample_idx < len(self.samples_indices)):
            raise IndexError("Sample index out of range.")

        sample_indices = self.samples_indices[sample_idx]

        # Get indices not in the sample (10% of the data)
        altered_prefs = self.prefs[~self.prefs.index.isin(sample_indices)]

        if prefs is None:
            prefs = self.prefs.copy()

        for index in altered_prefs.index:
            current_pref = prefs.loc[index, 'pref']
            possible_values = [value for value in [1, -1, 0] if value != current_pref]
            # possible_values = [value for value in [1, -1] if value != current_pref]
            rng = np.random.RandomState(seed=self.seed)
            new_pref = rng.choice(possible_values)
            prefs.loc[index, 'pref'] = new_pref


        return prefs



    def sample_items(self, count: int, seed: Optional[int] = 2338):
        """
        Sample a specified number of items and their corresponding preferences.
        :param count: Number of items to sample.
        :param seed: Seed for random number generator (optional).
        :return: A new PreferenceDataset object containing the sampled items and their preferences.
        """
        if seed is not None:
            np.random.seed(seed)
        sampled_item_ids = np.random.choice(self.items['item_id'], size=count, replace=False)
        sampled_items = self.items[self.items['item_id'].isin(sampled_item_ids)]
        sampled_prefs = self.prefs[(self.prefs['item_id_1'].isin(sampled_item_ids)) | (self.prefs['item_id_2'].isin(sampled_item_ids))]
        out = PreferenceDataset(prefs=sampled_prefs, items=sampled_items, judges=self.judges.copy(), unaltered_prefs = self.unaltered_prefs.copy())
        return out

    def sample_judges(self, count: int, seed: Optional[int] = 2338):
        """
        Sample a specified number of judges and their corresponding preferences.
        :param count: Number of judges to sample.
        :param seed: Seed for random number generator (optional).
        :return: A new PreferenceDataset object containing the sampled judges and their preferences.
        """
        if seed is not None:
            np.random.seed(seed)
        sampled_judge_ids = np.random.choice(self.judges['judge_id'], size=count, replace=False)
        sampled_judges = self.judges[self.judges['judge_id'].isin(sampled_judge_ids)]
        sampled_prefs = self.prefs[self.prefs['judge_id'].isin(sampled_judge_ids)]
        out = PreferenceDataset(prefs=sampled_prefs, items=self.items.copy(), judges=sampled_judges, preference_type= self.preference_type)
        return out

    def sample_preferences(self, method:str = 'random', count: int = 1000, seed = 2338, idx = 1):
        """
        Sample preferences with different strategies: randomly, by specific item, or by specific judge.
        :param method: Sampling method ('random', 'by_item', 'by_judge').
        :param count: Number of preferences to sample.
        :param item_id: Specific item ID for 'by_item' method.
        :param judge_id: Specific judge ID for 'by_judge' method.
        :return: A new PreferenceDataset object containing the sampled preferences.
        """
        assert method in ('random', 'by_item', 'by_judge'), "Unsupported sampling method."
        sampled_indices = set()
        if method == 'random':
            rng = np.random.RandomState(seed=seed * idx)
            sampled_indices.update(rng.choice(a=len(self.prefs), size=int(0.9 * len(self.prefs)), replace=False))
        # elif method == 'by_item':
        #     for item_id in self.items['item_id']:
        #         relevant_prefs = self.df[(self.df['item_id_1'] == item_id) | (self.df['item_id_2'] == item_id)]
        #         sampled_indices.update(np.random.choice(relevant_prefs.index, size=min(count, len(relevant_prefs)), replace=False))
        # elif method == 'by_judge':
        #     for judge_id in self.judges['judge_id']:
        #         relevant_prefs = self.df[self.df['judge_id'] == judge_id]
        #         sampled_indices.update(np.random.choice(relevant_prefs.index, size=min(count, len(relevant_prefs)), replace=False))
        # sampled_prefs = self.df.loc[list(sampled_indices)]
        return list(sampled_indices)

    #### File IO ####
    @staticmethod
    def read_csv(prefs_file, **kwargs):
        """
        Load a dataset of pairwise preferences from CSV files
        :param prefs_file: Path to CSV file of pairwise preferences
        :param judges_file: Path to CSV file of judges
        :param items_file: Path to CSV file of items
        :return: PreferenceDataset object
        """
        # Extract common file header from dataset file
        header = str(prefs_file).rsplit('_prefs.csv')[0]

        # Convert file names into path objects with the correct extension
        files = {
            'df': kwargs.get('prefs_file', f'{header}_prefs.csv'),
            'judges': kwargs.get('judges_file', f'{header}_judges.csv'),
            'items': kwargs.get('items_file', f'{header}_items.csv'),
            }
        assert Path(files['df']).is_file(), 'could not find preferences file'
        dfs = {k: pd.read_csv(f, sep=',', skipinitialspace=True) for k, f in files.items() if Path(f).is_file()}
        return PreferenceDataset(**dfs)

    def save(self, file, overwrite=True, check_save=True):
        """
        Save object to disk
        :param file: Path to the file where the dataset will be saved
        :param overwrite: Whether to overwrite the file if it already exists
        :param check_save: Whether to check if the saved file is equal to the original dataset
        :return: Path to the saved file
        """
        f = Path(file)
        if f.is_file() and not overwrite:
            raise IOError(f'file {f} already exists on disk')

        # Check data integrity
        self.__check_rep__()

        # Save a copy to disk
        data = copy(self)
        with open(f, 'wb') as outfile:
            dill.dump({'data': data}, outfile, protocol=dill.HIGHEST_PROTOCOL)

        if check_save:
            loaded = self.load(f)
            assert data == loaded
        return f

    @staticmethod
    def load(file):
        """
        Load processed data file from disk
        :param file: Path of the processed data file
        :return: data and cvindices
        """
        f = Path(file)
        if not f.is_file():
            raise IOError(f'file: {f} not found')

        with open(f, 'rb') as infile:
            file_contents = dill.load(infile)
            assert 'data' in file_contents, 'could not find `data` variable in pickle file contents'
            assert file_contents['data'].__check_rep__(), 'loaded `data` has been corrupted'

        return file_contents['data']

    #### Parsing functions ####
    @staticmethod
    def parse(file):
        """
        Parse a CSV file with pairwise preferences into a PreferenceDataset object
        :param file: Path to CSV file
        :return: PreferenceDataset object
        :raises: ValueError if the file type cannot be determined or is not supported

        The file name should contain one of the following substrings to indicate the type of preferences:
        - 'pairwise'
        - 'rankings'
        - 'ratings'

        Example file names:
        - 'dataset_pairwise.csv'
        - 'dataset_rankings.csv'
        - 'dataset_ratings.csv'
        """
        print(f'Parsing file: {file}')
        assert file.exists(), 'File does not exist'
        file_type = parse_preference_type(file.stem)
        if not file_type in PREFERENCE_TYPES:
            raise ValueError(f"Unknown preference type: {file_type}")
        parser = getattr(PreferenceDataset, f'parse_{file_type}')
        raw_df = pd.read_csv(file)
        out = parser(raw_df, parse_preference_type(file.stem))
        return out

    @staticmethod
    def create_preference_dictionaries(df):
        """
        Creates preference dictionaries from a DataFrame.
        Optimized for speed using vectorized operations.
        """

        # Use dictionaries directly since we populate all keys
        prefs_dict = {}
        judge_prefs = {}

        # Vectorized operations to count preferences
        mask_pref_1 = df['pref'] == 1
        item_pairs_pref_1 = list(zip(df.loc[mask_pref_1, 'item_id_1'], df.loc[mask_pref_1, 'item_id_2']))
        item_pairs_pref_neg_1 = list(zip(df.loc[~mask_pref_1, 'item_id_2'], df.loc[~mask_pref_1, 'item_id_1']))

        for item_pair in item_pairs_pref_1:
            prefs_dict[item_pair] = prefs_dict.get(item_pair, 0) + 1

        for item_pair in item_pairs_pref_neg_1:
            prefs_dict[item_pair] = prefs_dict.get(item_pair, 0) + 1

        # Vectorized operations to calculate judge preferences
        for judge_id, item_id_1, item_id_2, pref in zip(df['judge_id'], df['item_id_1'], df['item_id_2'], df['pref']):
            if pref == 1:
                judge_prefs[(judge_id, item_id_1, item_id_2)] = 1
            elif pref == -1:
                judge_prefs[(judge_id, item_id_2, item_id_1)] = 1

        return prefs_dict, judge_prefs
    @staticmethod
    def parse_rankings(raw_df, preference_type, intermediate_dir="intermediate_results"):
        """
        Parse a CSV file with rankings into a PreferenceDataset object
        :param raw_df: Raw DataFrame with rankings
        :param preference_type: The type of preference (e.g., ranking or rating)
        :param intermediate_dir: Directory to save intermediate results to disk
        :return: PreferenceDataset object
        :raises: ValueError if the input DataFrame format is incorrect

        Input DataFrame format:
        - The first column should be 'item_name', containing the names of the items.
        - The subsequent columns should be named after the judges and contain the rankings given by each judge.
        - The rankings should be numeric values starting from 1.

        Example CSV:
        item_name,judge_1,judge_2,judge_3
        item_A,3,2,1
        item_B,1,3,2
        item_C,2,1,3
        """

        if raw_df.shape[1] < 2:
            raise ValueError("CSV format error: At least two columns are required (item_name, and at least one judge).")

        headers = raw_df.columns.to_list()
        assert 'item_name' in headers, "CSV format error: Missing 'item_name' column."

        item_names = raw_df['item_name'].tolist()
        assert len(set(item_names)) == len(item_names), "Item names are not unique."
        n_items = len(item_names)
        assert n_items >= 1

        judge_names = [name for name in headers if name != 'item_name']
        assert len(set(judge_names)) == len(judge_names), "Judge names are not unique."
        n_judges = len(judge_names)
        assert n_judges >= 1

        # Create items DataFrame
        items_df = pd.DataFrame({
            'item_id': range(1, n_items + 1),
            'item_name': item_names
        })

        # Create judges DataFrame
        judges_df = pd.DataFrame({
            'judge_id': range(1, n_judges + 1),
            'judge_name': judge_names
        })

        # Merge item IDs to judge rankings once, avoiding repetitive lookups
        raw_df = pd.merge(raw_df, items_df[['item_name', 'item_id']], on='item_name')

        # Ensure intermediate directory exists
        intermediate_path = Path(intermediate_dir)
        if not intermediate_path.exists():
            intermediate_path.mkdir(parents=True, exist_ok=True)

        prefs_list = []
        batch_prefs = []

        # Iterate over each judge to generate preference pairs and save intermediate results every ~50 judges
        for judge_id, judge_name in enumerate(judge_names, start=1):
            print(f"Processing judge_id: {judge_id}")
            judge_ratings = raw_df[['item_id', judge_name]].copy().values
            item_ids = judge_ratings[:, 0]
            ranks = judge_ratings[:, 1]

            # Use combinations to get all unique pairs (no upper/lower restriction)
            item_id_pairs = np.array(list(combinations(item_ids, 2)))

            # Extract ranks for each pair efficiently using NumPy indexing
            rank1 = ranks[np.searchsorted(item_ids, item_id_pairs[:, 0])]
            rank2 = ranks[np.searchsorted(item_ids, item_id_pairs[:, 1])]
            prefs = np.where(np.isnan(rank1) | np.isnan(rank2), np.nan,
                             np.where(rank1 == rank2, 0, np.where(rank1 > rank2, 1, -1)))

            # Append preferences for each pair to batch
            batch_prefs.extend([
                {
                    'judge_id': judge_id,
                    'item_id_1': np.minimum(item1, item2),
                    'item_id_2': np.maximum(item1, item2),
                    'pref': -pref if item1 < item2 else pref
                }
                for (item1, item2), pref in zip(item_id_pairs, prefs)
            ])

            # Save intermediate results every 50 judges and remove from memory
            if judge_id % 25 == 0 or judge_id == n_judges:
                intermediate_file = Path(intermediate_dir) / f"judges_{judge_id - 49}_to_{judge_id}_prefs.csv"
                pd.DataFrame(batch_prefs).to_csv(intermediate_file, index=False)
                prefs_list.append(intermediate_file)
                batch_prefs = []

        # Load all intermediate results from disk and concatenate
        prefs_df = pd.concat([pd.read_csv(file) for file in prefs_list], ignore_index=True)
        prefs_df = prefs_df.sort_values(by=['judge_id', 'item_id_1', 'item_id_2']).reset_index(drop=True)

        dfs = {
            'items': items_df,
            'judges': judges_df,
            'prefs': prefs_df,
            'preference_type': preference_type,
        }

        # Assuming create_preference_dictionaries is efficient enough and left unchanged
        judge_prefs, prefs_dict = PreferenceDataset.create_preference_dictionaries(prefs_df)
        dfs['prefs_dict'] = prefs_dict
        dfs['judge_prefs'] = judge_prefs

        return PreferenceDataset(**dfs)

    @staticmethod
    def parse_ratings(raw_df, preference_type, intermediate_dir="intermediate_results"):
        """
        Parse a CSV file with ratings into a PreferenceDataset object
        :param raw_df: Raw DataFrame with ratings
        :param preference_type: The type of preference (e.g., ranking or rating)
        :param intermediate_dir: Directory to save intermediate results to disk
        :return: PreferenceDataset object
        :raises: ValueError if the input DataFrame format is incorrect

        Input DataFrame format:
        - The first column should be 'item_name', containing the names of the items.
        - The subsequent columns should be named after the judges and contain the ratings given by each judge.
        - The ratings should be numeric values.

        Example CSV:
        item_name,judge_1,judge_2,judge_3
        item_A,4.5,3.0,5.0
        item_B,3.0,4.0,2.5
        item_C,4.0,5.0,3.5
        """

        if raw_df.shape[1] < 2:
            raise ValueError("CSV format error: At least two columns are required (item_name, and at least one judge).")

        headers = raw_df.columns.to_list()
        assert 'item_name' in headers, "CSV format error: Missing 'item_name' column."

        item_names = raw_df['item_name'].tolist()
        assert len(set(item_names)) == len(item_names), "Item names are not unique."
        n_items = len(item_names)
        assert n_items >= 1

        judge_names = [name for name in headers if name != 'item_name']
        assert len(set(judge_names)) == len(judge_names), "Judge names are not unique."
        n_judges = len(judge_names)
        assert n_judges >= 1

        # Create items DataFrame
        items_df = pd.DataFrame({
            'item_id': range(1, n_items + 1),
            'item_name': item_names
        })

        # Create judges DataFrame
        judges_df = pd.DataFrame({
            'judge_id': range(1, n_judges + 1),
            'judge_name': judge_names
        })

        # Merge item IDs to judge ratings once, avoiding repetitive lookups
        raw_df = pd.merge(raw_df, items_df[['item_name', 'item_id']], on='item_name')

        # Ensure intermediate directory exists
        intermediate_path = Path(intermediate_dir)
        if not intermediate_path.exists():
            intermediate_path.mkdir(parents=True, exist_ok=True)

        prefs_list = []
        batch_prefs = []

        # Iterate over each judge to generate preference pairs and save intermediate results every ~50 judges
        for judge_id, judge_name in enumerate(judge_names, start=1):
            print(f"Processing judge_id: {judge_id}")
            judge_ratings = raw_df[['item_id', judge_name]].copy().values
            item_ids = judge_ratings[:, 0]
            ratings = judge_ratings[:, 1]

            # Use combinations to get all unique pairs (no upper/lower restriction)
            item_id_pairs = np.array(list(combinations(item_ids, 2)))

            # Extract ratings for each pair efficiently using NumPy indexing
            rating1 = ratings[np.searchsorted(item_ids, item_id_pairs[:, 0])]
            rating2 = ratings[np.searchsorted(item_ids, item_id_pairs[:, 1])]
            prefs = np.where(np.isnan(rating1) | np.isnan(rating2), np.nan,
                             np.where(rating1 == rating2, 0, np.where(rating1 > rating2, 1, -1)))

            # Append preferences for each pair to batch
            batch_prefs.extend([
                {
                    'judge_id': judge_id,
                    'item_id_1': np.minimum(item1, item2),
                    'item_id_2': np.maximum(item1, item2),
                    'pref': pref if item1 < item2 else -pref
                }
                for (item1, item2), pref in zip(item_id_pairs, prefs)
            ])

            # Save intermediate results every 50 judges and remove from memory
            if judge_id % 25 == 0 or judge_id == n_judges:
                intermediate_file = Path(intermediate_dir) / f"judges_{judge_id - 24}_to_{judge_id}_prefs.csv"
                pd.DataFrame(batch_prefs).to_csv(intermediate_file, index=False)
                prefs_list.append(intermediate_file)
                batch_prefs = []

        # Load all intermediate results from disk and concatenate
        prefs_df = pd.concat([pd.read_csv(file) for file in prefs_list], ignore_index=True)
        prefs_df = prefs_df.sort_values(by=['judge_id', 'item_id_1', 'item_id_2']).reset_index(drop=True)

        dfs = {
            'items': items_df,
            'judges': judges_df,
            'prefs': prefs_df,
            'preference_type': preference_type,
        }
        print(f"Creating preference dictionaries")

        # Assuming create_preference_dictionaries is efficient enough and left unchanged
        judge_prefs, prefs_dict = PreferenceDataset.create_preference_dictionaries(prefs_df)
        dfs['prefs_dict'] = prefs_dict
        dfs['judge_prefs'] = judge_prefs

        print(f"Preference dictionaries created")

        return PreferenceDataset(**dfs)

    def parse_ratings_expert(raw_df, preference_type, intermediate_dir="intermediate_results"):
        if raw_df.shape[1] < 3:
            raise ValueError(
                "CSV format error: At least three columns are required (item_name, at least one judge, and expert).")

        headers = raw_df.columns.to_list()
        assert 'item_name' in headers, "CSV format error: Missing 'item_name' column."
        assert 'expert' in headers, "CSV format error: Missing 'expert' column."

        item_names = raw_df['item_name'].tolist()
        assert len(set(item_names)) == len(item_names), "Item names are not unique."
        n_items = len(item_names)
        assert n_items >= 1

        judge_names = [name for name in headers if name not in ['item_name', 'expert']]
        assert len(set(judge_names)) == len(judge_names), "Judge names are not unique."
        n_judges = len(judge_names)
        assert n_judges >= 1

        items_df = pd.DataFrame({'item_id': range(1, n_items + 1), 'item_name': item_names})
        judges_df = pd.DataFrame({'judge_id': range(1, n_judges + 1), 'judge_name': judge_names})

        raw_df = pd.merge(raw_df, items_df[['item_name', 'item_id']], on='item_name')
        intermediate_path = Path(intermediate_dir)
        intermediate_path.mkdir(parents=True, exist_ok=True)

        expert_ratings = dict(zip(raw_df['item_id'], raw_df['expert']))
        prefs_list = []
        batch_prefs = []

        for judge_id, judge_name in enumerate(judge_names, start=1):
            print(f"Processing judge_id: {judge_id}")
            judge_ratings = raw_df[['item_id', judge_name]].copy().values
            item_ids = judge_ratings[:, 0]
            ratings = judge_ratings[:, 1]
            item_id_pairs = np.array(list(combinations(item_ids, 2)))
            rating1 = ratings[np.searchsorted(item_ids, item_id_pairs[:, 0])]
            rating2 = ratings[np.searchsorted(item_ids, item_id_pairs[:, 1])]
            prefs = np.where(np.isnan(rating1) | np.isnan(rating2), np.nan,
                             np.where(rating1 > rating2, 1, np.where(rating1 < rating2, -1, np.nan)))

            for i, ((item1, item2), pref) in enumerate(zip(item_id_pairs, prefs)):
                if np.isnan(pref):
                    expert_rating1, expert_rating2 = expert_ratings[item1], expert_ratings[item2]
                    if np.isnan(expert_rating1) or np.isnan(expert_rating2):
                        continue
                    pref = 0 if expert_rating1 == expert_rating2 else (1 if expert_rating1 > expert_rating2 else -1)

                batch_prefs.append({
                    'judge_id': judge_id,
                    'item_id_1': min(item1, item2),
                    'item_id_2': max(item1, item2),
                    'pref': pref if item1 < item2 else -pref
                })

            if judge_id % 25 == 0 or judge_id == n_judges:
                intermediate_file = intermediate_path / f"judges_{judge_id - 24}_to_{judge_id}_prefs.csv"
                pd.DataFrame(batch_prefs).to_csv(intermediate_file, index=False)
                prefs_list.append(intermediate_file)
                batch_prefs = []

        prefs_df = pd.concat([pd.read_csv(file) for file in prefs_list], ignore_index=True)
        prefs_df = prefs_df.sort_values(by=['judge_id', 'item_id_1', 'item_id_2']).reset_index(drop=True)

        dfs = {
            'items': items_df,
            'judges': judges_df,
            'prefs': prefs_df,
            'preference_type': preference_type,
        }

        judge_prefs, prefs_dict = PreferenceDataset.create_preference_dictionaries(prefs_df)
        dfs['prefs_dict'] = prefs_dict
        dfs['judge_prefs'] = judge_prefs

        return PreferenceDataset(**dfs)

    @staticmethod
    def parse_pairwise(raw_df, preference_type):
        """
        Parse a CSV file with pairwise preferences into a PreferenceDataset object
        :param raw_df: Raw DataFrame with pairwise preferences
        :param preference_type: Type of preference data (e.g., 'pairwise')
        :return: PreferenceDataset object
        :raises: ValueError if the input DataFrame format is incorrect

        Input DataFrame format:
        - The DataFrame should contain four columns: 'judge_id', 'item_id_1', 'item_id_2', and 'pref'.
        - 'judge_id': ID of the judge.
        - 'item_id_1': ID of the first item.
        - 'item_id_2': ID of the second item.
        - 'pref': Preference value (1 for preferring item_1 over item_2, -1 for the opposite, 0 for no preference, or np.nan for missing preference).

        Example CSV:
        judge_id,item_id_1,item_id_2,pref
        1,101,102,1
        1,101,103,-1
        2,101,102,0
        3,101,102,
        """

        required_columns = {'judge_id', 'item_id_1', 'item_id_2', 'pref'}
        if not required_columns.issubset(raw_df.columns):
            raise ValueError(f"CSV format error: Missing one or more required columns: {required_columns}")

        # Get unique item and judge IDs from the pairwise preference data
        unique_item_ids = pd.unique(pd.concat([raw_df['item_id_1'], raw_df['item_id_2']]))
        unique_judge_ids = raw_df['judge_id'].unique()
        unique_item_names = unique_item_ids

        if "item_name_one" and "item_name_two" in raw_df.columns:
            unique_item_names = pd.unique(pd.concat([raw_df['item_name_one'], raw_df['item_name_two']]))

        # Create DataFrame for items and judges
        items_df = pd.DataFrame({
            'item_id': unique_item_ids,
            'item_name': unique_item_names
        }).reset_index(drop=True)

        judges_df = pd.DataFrame({
            'judge_id': unique_judge_ids,
            'judge_name': unique_judge_ids
        }).reset_index(drop=True)

        # Validate preference values, allowing for NaN
        valid_preference_values = {1, -1, 0, np.nan}
        if not set(raw_df['pref'].dropna()).issubset({1, -1, 0}):
            raise ValueError(
                f"Invalid preference values detected. Only 1, -1, np.nan, and 0 are allowed in 'pref' column.")

        # Generate preference dictionaries
        judge_prefs, prefs_dict = PreferenceDataset.create_preference_dictionaries(raw_df)

        # Create and return the PreferenceDataset object
        return PreferenceDataset(
            prefs=raw_df,
            items=items_df,
            judges=judges_df,
            preference_type=preference_type,
            prefs_dict=prefs_dict,
            judge_prefs=judge_prefs
        )

    #### Helper functions to handle rank tables ####
    @staticmethod
    def check_rank_table(df):
        """
        Check correctness of a rank table
        :param df: DataFrame containing rank data
        :return: True if the rank table is valid
        """
        assert isinstance(df, pd.DataFrame), 'rank table should be a pd.DataFrame'
        assert 'rank' in df, "rank table should contain a column named 'rank'"
        assert df.groupby(['item_id', 'judge_id']).size().eq(1).all(), "rank table should have one rank per (item_id, judge_id)"
        assert df['rank'].ge(1).all(), "df should be >= 1"
        assert np.array_equal(df['rank'], np.require(df['rank'], np.int_)), 'df should be integer-valued'
        grouped = df.groupby(['judge_id'])
        assert grouped['rank'].min().eq(1).all(), "rank does not start from 1 for some judge"
        assert grouped['rank'].nunique().eq(grouped['rank'].max() - grouped['rank'].min() + 1).all(), "ranks are not sequential for some judges"
        return True

    @staticmethod
    def ranks_to_prefs(ranks):
        """
        Split a table of ranks into a table of pairwise preferences
        :param ranks: pd.DataFrame with [judge_id, item_id, rank]
        :return: df: pd.DataFrame with [judge_id, item_id_1, item_id_2, pref]
        """
        PreferenceDataset.check_rank_table(ranks)

        df = pd.DataFrame(ranks)
        df.set_index('judge_id', inplace=True)
        df = df.join(df, how="inner", lsuffix="_1", rsuffix="_2")
        keep_idx = df['item_id_1'] < df['item_id_2']
        df = df[keep_idx]

        df['pref'] = float('nan')
        df.loc[df['rank_1'] < df['rank_2'], 'pref'] = PREFERENCE_VALUES['prefer_1']
        df.loc[df['rank_2'] < df['rank_1'], 'pref'] = PREFERENCE_VALUES['prefer_2']
        df.loc[df['rank_2'] == df['rank_1'], 'pref'] = PREFERENCE_VALUES['prefer_none']
        assert not df['pref'].isna().any()

        df['pref'] = df['pref'].astype(int)
        df.drop(['rank_1', 'rank_2'], axis=1, inplace=True)
        df.reset_index(inplace=True)
        return df