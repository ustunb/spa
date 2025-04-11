from dataclasses import InitVar, dataclass, field
import time
import networkx as nx
import numpy as np
import pandas as pd
from spa.data import PREFERENCE_VALUES, PreferenceDataset
from spa.ranking import Ranking
from tqdm.auto import tqdm

def to_ranking_df(ordered_components):
    """
    converts strongly connected components (list of strings from topological sort)
    into a pandas dataframe of tiers and items
    """
    parsed = {
        "rank": [],
        "item_id": [],
    }
    # ordered_components = [str(ordered_components.nodes[i]['members']) for i in ordered_components]
    # convert from a string formatted as "{'A', 'B', 'C'}" into array of strings ['A', 'B', 'C']
    for tier_id, component in enumerate(ordered_components):
        for item in component:
            parsed['rank'].append(tier_id + 1)
            parsed['item_id'].append(int(float(item)))

    df = pd.DataFrame.from_dict(parsed).astype({
        'rank': 'int32',
        'item_id': 'int32',
    })
    return df

from spa.debug import ipsh
class SelectiveRankAggregator:

    def __init__(self, data, prefs=None, items=None, adjust_weights = False, equalize_weights = False, ignore_ties = True, ignore_missing = False, exceed_max_dissent = False, ties_as_half = False):
        """
        :param data: A dataset containing preference information.
        """
        assert isinstance(data, PreferenceDataset)
        self.n_judges = data.n_judges
        self.id_to_name = data.id_to_name
        self.prefs = data.prefs if prefs is None else prefs
        self.items = data.items if items is None else items
        self.adjust_weights = adjust_weights
        self.ignore_missing = ignore_missing
        self.exceed_max_dissent = exceed_max_dissent
        self.ignore_ties = ignore_ties

        # create graph
        self.edge_weights = self.calculate_edge_weights()
        G = nx.DiGraph()

        if self.adjust_weights:
            print("Adjusting weights")
            for _, edge in self.edge_weights.iterrows():
                G.add_edge(edge['source_id'], edge['dest_id'], weight = edge['v_ij'])
                G.add_edge(edge['dest_id'], edge['source_id'], weight = edge['v_ji'])

        elif equalize_weights and ties_as_half:
            print("Equalizing weights and ties as half")
            for _, edge in self.edge_weights.iterrows():
                sum_weights = edge['prefer_1'] + edge['prefer_2'] + edge['prefer_none']
                w_ij = edge['prefer_1'] + 0.5 * edge['prefer_none'] / sum_weights * 100
                w_ji = edge['prefer_2'] + 0.5 * edge['prefer_none'] / sum_weights * 100
                #set edge weights
                edge['w_ij'] = w_ij
                edge['w_ji'] = w_ji
                G.add_edge(edge['source_id'], edge['dest_id'], weight = w_ij)
                G.add_edge(edge['dest_id'], edge['source_id'], weight = w_ji)

        elif equalize_weights and ignore_ties is False:
            print("Equalizing weights with ties")
            for _, edge in self.edge_weights.iterrows():
                sum_weights = edge['prefer_1'] + edge['prefer_2'] + 2 * edge['prefer_none']
                w_ij = edge['prefer_1'] + edge['prefer_none'] / sum_weights * 100
                w_ji = edge['prefer_2'] + edge['prefer_none'] / sum_weights * 100
                #set edge weights
                edge['w_ij'] = w_ij
                edge['w_ji'] = w_ji
                G.add_edge(edge['source_id'], edge['dest_id'], weight = w_ij)
                G.add_edge(edge['dest_id'], edge['source_id'], weight = w_ji)

        elif equalize_weights:
            print("Equalizing weights WITHOUT ties")
            for _, edge in self.edge_weights.iterrows():
                sum_weights = edge['prefer_1'] + edge['prefer_2']
                w_ij = edge['prefer_1'] / sum_weights * 100
                w_ji = edge['prefer_2'] / sum_weights * 100
                #set edge weights
                edge['w_ij'] = w_ij
                edge['w_ji'] = w_ji
                G.add_edge(edge['source_id'], edge['dest_id'], weight = w_ij)
                G.add_edge(edge['dest_id'], edge['source_id'], weight = w_ji)

        elif self.ignore_ties and ties_as_half:
            for _, edge in self.edge_weights.iterrows():
                prefer_1 = edge['prefer_1'] + 0.5 * edge['prefer_none']
                prefer_2 = edge['prefer_2'] + 0.5 * edge['prefer_none']
                G.add_edge(edge['source_id'], edge['dest_id'], weight = prefer_1)
                G.add_edge(edge['dest_id'], edge['source_id'], weight = prefer_2)
        elif ties_as_half:
            print("Ties as half")
            for _, edge in self.edge_weights.iterrows():
                prefer_1 = edge['prefer_1'] + 0.5 * edge['prefer_none'] + edge['prefer_missing']
                prefer_2 = edge['prefer_2'] + 0.5 * edge['prefer_none'] + edge['prefer_missing']
                G.add_edge(edge['source_id'], edge['dest_id'], weight = prefer_1)
                G.add_edge(edge['dest_id'], edge['source_id'], weight = prefer_2)
        elif self.ignore_ties:
            print("Ignoring ties")
            for _, edge in self.edge_weights.iterrows():
                G.add_edge(edge['source_id'], edge['dest_id'], weight = edge['prefer_1'])
                G.add_edge(edge['dest_id'], edge['source_id'], weight = edge['prefer_2'])
        else:
            print("Normal Weighting")
            for _, edge in self.edge_weights.iterrows():
                G.add_edge(edge['source_id'], edge['dest_id'], weight = edge['w_ij'])
                G.add_edge(edge['dest_id'], edge['source_id'], weight = edge['w_ji'])
        self.G = G

        # setup dissent values
        if self.adjust_weights:
            dissent_values = np.unique(np.concatenate((self.edge_weights['v_ij'].values, self.edge_weights['v_ji'].values)))
        else:
            dissent_values = np.unique(np.concatenate((self.edge_weights['w_ij'].values, self.edge_weights['w_ji'].values)))


        dissent_max = self.n_judges / 2.0
        if dissent_max.is_integer():
            dissent_max = int(dissent_max)
        else:
            dissent_max = np.floor(self.n_judges / 2.0) + 1
        if not self.exceed_max_dissent:
            dissent_values = dissent_values[np.less(dissent_values, dissent_max)]

        self.all_dissent_rates = dissent_values / self.n_judges

    def calculate_edge_weights(self):
        """
        Calculate the edge weights for each pair of items in the graph.
        """
        df = pd.DataFrame(self.prefs)
        df = df.rename(columns={'item_id_1': 'source_id', 'item_id_2': 'dest_id'})
        df.pref.replace(to_replace={v: k for k, v in PREFERENCE_VALUES.items()}, inplace=True)
        df = df.groupby(['source_id', 'dest_id', 'pref']).size().reset_index(name='weight')
        df = df.pivot_table(index=['source_id', 'dest_id'], columns='pref', values='weight').reset_index()
        df = df.rename_axis('pair_id', axis=1)
        # Handle missing preference columns
        missing_columns = list(set(PREFERENCE_VALUES.keys()).difference(df.columns))
        df[missing_columns] = float('nan')
        df = df.fillna(0.0)
        df['prefer_missing'] = self.n_judges - df['prefer_1'] - df['prefer_2'] - df['prefer_none']

        # Add edge weight columns
        df['w_ij'] = df['prefer_1'] + df['prefer_none']
        df['w_ji'] = df['prefer_2'] + df['prefer_none']
        df['adjustment'] = self.n_judges / (self.n_judges - df['prefer_missing'])
        df['v_ij'] = (df['w_ij'] * self.n_judges) / (self.n_judges - df['prefer_missing'])
        df['v_ji'] = (df['w_ji'] * self.n_judges) / (self.n_judges - df['prefer_missing'])
        if self.ignore_missing is False:
            df['w_ij'] += df['prefer_missing']
            df['w_ji'] += df['prefer_missing']

        n = len(self.items)
        assert all(df[['prefer_1', 'prefer_2', 'prefer_missing', 'prefer_none']].sum(axis = 1) == self.n_judges)
        assert df[df.isna().any(axis = 1)].empty
        assert df.shape[0] == n * (n - 1) / 2
        return df

    def fit(self, dissent_rate, return_df=False, **kwargs):
        start_time = time.time()

        G = self.G.copy()
        edges = self.filter_graph(G, dissent_rate)
        G.remove_edges_from(edges)

        try:
            condensed_graph = nx.condensation(G)
            topologically_sorted = list(nx.topological_sort(condensed_graph))
            ordered_components = [set(condensed_graph.nodes[i]['members']) for i in topologically_sorted]

            ranking, df = self.create_ranking_df(ordered_components, return_df)

        except nx.NetworkXUnfeasible:
             ranking, df = None, None

        runtime = time.time() - start_time

        if return_df:
            return ranking, df, runtime
        else:
            return ranking, runtime

    def fit_path(self, dissent_rates=None, return_df=False, lower_bound = 0, upper_bound = 1, **kwargs):
        """
        Fits the model for different dissent rates and returns a concatenated path of rankings.
        Tracks the runtime for each dissent rate and returns them as an array.

        Returns:
            RankingPath: A concatenated path of distinct rankings across different dissent rates.
            runtimes: An array of runtimes for each dissent rate.
        """
        if dissent_rates is not None:
            path_rates = np.array(dissent_rates)
            for r in path_rates:
                assert check_dissent_rate(r)
            path_rates = np.unique(path_rates)
        else:
            path_rates = self.all_dissent_rates

        G = self.G.copy()
        outputs = {}
        runtimes = []
        all_dropped_edges = []
        len_ordered_components = -1

        if len(path_rates) <= 1:
            return None

        if 0 not in path_rates and upper_bound < 1:
            path_rates = np.concatenate(([0], path_rates))

        for dissent_rate in tqdm(path_rates):
            if dissent_rate < lower_bound or dissent_rate > upper_bound:
                continue

            start_time = time.time()
            edges = self.filter_graph(G, dissent_rate)
            all_dropped_edges.extend(edges)
            disconnected = any([(v, u) in all_dropped_edges for u, v, d in edges])
            if disconnected:
                # print disconnect message
                print(f"graph disconnect at dissent: {dissent_rate}.")
                for (u, v, weight) in edges:
                    if (v, u) in all_dropped_edges:
                        print(f"No edge between: {u} and {v}, weight: {weight}")
                # check lengths
                # set([len(v) for k, v in outputs.items()])
                runtimes.append(time.time() - start_time)
                break
            G.remove_edges_from(edges)
            condensed_graph = nx.condensation(G)
            topologically_sorted = list(nx.topological_sort(condensed_graph))
            ordered_components = [set(condensed_graph.nodes[i]['members']) for i in topologically_sorted]
            if len(ordered_components) != len_ordered_components:
                print("Unique rankings found at dissent rate: ", dissent_rate)
                len_ordered_components = len(ordered_components)
                ranking, df = self.create_ranking_df(ordered_components, return_df)
                outputs[dissent_rate] = (ranking, df) if return_df else ranking

            runtimes.append(time.time() - start_time)

        print(outputs)
        out = RankingPath(outputs)
        return out, runtimes

    def filter_graph(self, G, dissent_rate):
        dissent_threshold = dissent_rate * self.n_judges
        edges_to_remove = [(u, v, d["weight"]) for u, v, d in G.edges(data=True) if d["weight"] <= dissent_threshold]
        return edges_to_remove

    def create_ranking_df(self, ordered_components, return_df):
        """
        Create a ranking from ordered components.
        :param ordered_components: List of topologically sorted components.
        :param return_df: Whether to return DataFrame with the ranking.
        :return: Ranking object and optional DataFrame.
        """
        # Create ranking DataFrame from ordered components
        df = to_ranking_df(ordered_components)
        df['item_name'] = df['item_id'].map(self.id_to_name)
        ranking = Ranking(df)
        out = (ranking, df) if return_df else (ranking, None)
        return out


@dataclass
class RankingPath:
    """
    Note: this only runs correctly if it is initialized with a set of all possible dissent values between (0,0.5)
    """
    raw: InitVar[dict]
    rankings: dict = field(default_factory=dict)

    def __post_init__(self, raw: dict):
        assert len(raw) > 0
        ordered_rates = sorted(raw.keys())
        base_ranking = raw[ordered_rates[0]]
        rankings = {}
        for k in ordered_rates:
            assert check_dissent_rate(k), f"Invalid dissent value: {k}"
            ranking = raw[k]
            assert isinstance(ranking, Ranking), f"Invalid ranking for dissent value {k}"
            assert ranking.is_comparable_to(
                base_ranking), f"Ranking for dissent value {k} is not comparable to base ranking"
            rankings[k] = ranking

        self.rankings = rankings
        self.__check_rep__()

    def __getitem__(self, dissent_rate):
        """
        Retrieves the ranking for a given dissent value.
        :param dissent_rate: The dissent value to retrieve the ranking for.
        :return: The Ranking object.
        """
        assert check_dissent_rate(dissent_rate)
        if dissent_rate in self.dissent_rates:
            return self.rankings[dissent_rate]
        else:
            idx = np.searchsorted(self.dissent_rates, dissent_rate, side='right')
            if idx == len(self):
                val = self.dissent_rates[idx - 1]
            else:
                val = self.dissent_rates[idx]
            out = self.rankings.get(val)
            return out

    def __setitem__(self, dissent_rate, ranking):
        """
        Adds a ranking to the path.
        :param dissent_rate: The dissent value associated with the ranking.
        :param ranking: The Ranking object.
        """
        assert check_dissent_rate(dissent_rate)
        assert isinstance(ranking, Ranking)
        if dissent_rate in self.rankings:
            raise ValueError(f"Ranking for dissent value {dissent_rate} already exists.")
        current_ranking = self[dissent_rate]

        assert ranking.is_comparable_to(
            current_ranking), f"Ranking for dissent value {dissent_rate} is not comparable to existing ranking"
        if current_ranking != ranking:
            self.rankings[dissent_rate] = ranking

    @property
    def dissent_rates(self):
        out = np.array(sorted(self.rankings.keys()))
        return out

    def __len__(self):
        return len(self.rankings)

    def __repr__(self):
        return f"RankingPath({self.dissent_rates})"

    def __check_rep__(self):
        """
        Checks if the preferences are consistent across all dissent values.
        :return: Tuple (True if preferences are consistent, False otherwise, Number of failures, List of failure pairs).
        """
        if len(self) < 2:
            return True, 0, []

    def __repr__(self):
        return f"RankingPath({self.dissent_rates})"

    def __check_rep__(self):
        """
        Checks if the preferences are consistent across all dissent values.
        :return: Tuple (True if preferences are consistent, False otherwise, Number of failures, List of failure pairs).
        """
        if len(self) < 2:
            return True, 0, []

        for a in self.dissent_rates:
            R = self[a]
            larger_rates = [r for r in self.dissent_rates if r > a]
            for b in larger_rates:
                S = self[b]
                #todo: make this work with the new setup

                # if not R.is_compatible_with(S):
                #     raise ValueError(f"Rankings for dissent values {a} and {b} are not comparable")
                # return False

        return True


def check_dissent_rate(rate):
    assert np.isfinite(rate), "dissent must be finite"
    assert np.greater_equal(rate, 0.0), "dissent must be a float between 0 and 1"
    # removed check for experitmation
    # assert np.less(rate, 0.5), "dissent must be less than 0.5"
    return True