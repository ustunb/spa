import os
import statistics
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
import fnmatch
import dill
import numpy as np
import pandas as pd
from psutil import Process
from spa.data import PreferenceDataset
from spa.paths import get_processed_data_file, results_dir

settings = {
    # "data_names": ["nba_coach_2021", "survivor", "law_modified", "sushi", "csrankings", "btl", "btl_modified", "airlines"],
    "data_names": ["synthetic100"],
    "method_names": ["spa"],
    "seed": 2338,
    "n_samples": 100,
    "enforce_sample_size": True,
    "save_results": True,
    "modify_existing": False
}

# Argument parser to get input from the command line
if Process(pid=os.getppid()).name() not in ("pycharm"):
    p = ArgumentParser()
    p.add_argument('-d', '--data_names', type=str, nargs='+', default=settings['data_names'], help='name of dataset(s)')
    p.add_argument('-m', '--method_names', type=str, nargs='+', default=settings['method_names'], help='name of method(s)')
    p.add_argument('-s', '--seed', type=int, default=settings['seed'], help='seed for random number generation'),
    p.add_argument('-n', '--n_samples', type=int, default=settings['n_samples'], help='number of samples to generate')
    p.add_argument('-f', '--enforce_sample_size', action='store_true', default=settings['enforce_sample_size'], help='enforce sample size')
    p.add_argument('--modify_existing', action='store_true', default=settings['modify_existing'], help='modify existing results file')
    args, _ = p.parse_known_args()
    settings.update(vars(args))


def load_preferences(prefs, include_judge_id=False):
    """Loads preferences into a dictionary, handling NaN values."""
    prefs_dict = {}

    for _, row in prefs.iterrows():
        pref_value = row['pref']
        pref_value = 0 if pd.isna(pref_value) else int(float(pref_value))

        item_id_1 = int(float(row['item_id_1']))
        item_id_2 = int(float(row['item_id_2']))

        if include_judge_id:
            judge_id = int(float(row['judge_id']))
            key = (item_id_1, item_id_2, judge_id)
        else:
            key = (item_id_1, item_id_2)

        prefs_dict[key] = pref_value
        prefs_dict[(key[1], key[0]) if len(key) == 2 else (key[1], key[0], key[2])] = -pref_value

    return prefs_dict

def calculate_comparison_metrics(judge_prefs, ranking_prefs):
    """Calculates comparison metrics between judge and ranking preferences."""
    metrics = {
        'disagreement_count': 0,
        'total_disagreements_pairwise': 0,
        'disagreement': {},
    }
    tracked_pairs = set()

    for (item_1, item_2, judge_id), value in judge_prefs.items():
        if value == 0 or (item_1, item_2, judge_id) in tracked_pairs:
            continue

        tracked_pairs.update({(item_1, item_2, judge_id), (item_2, item_1, judge_id)})
        metrics['disagreement'][judge_id] = metrics['disagreement'].get(judge_id, 0)

        if (item_1, item_2) in ranking_prefs and ranking_prefs[(item_1, item_2)] != 0 and value == ranking_prefs[(item_1, item_2)]:
            metrics['disagreement_count'] += 1
            metrics['total_disagreements_pairwise'] += 1
            metrics['disagreement'][judge_id] += 1
        elif (item_2, item_1) in ranking_prefs and ranking_prefs[(item_2, item_1)] != 0 and -value == ranking_prefs[(item_2, item_1)]:
            metrics['disagreement_count'] += 1
            metrics['total_disagreements_pairwise'] += 1
            metrics['disagreement'][judge_id] += 1

    if metrics['disagreement']:
        metrics['min_disagreement_per_judge_pairwise'] = min(metrics['disagreement'].values())
        metrics['max_disagreement_per_judge_pairwise'] = max(metrics['disagreement'].values())
        metrics['med_disagreement_per_judge_pairwise'] = statistics.median(metrics['disagreement'].values())

    del metrics['disagreement']
    return metrics

def calculate_ranking_metrics(data, ranking, runtime_metrics):
    """Calculates ranking-related metrics and includes runtime metrics."""
    metrics = {
        'n_items_cnt': len(ranking.items),
        'missing_pairwise': round(np.isnan(data.prefs["pref"]).sum() / len(data.prefs["pref"]), 3),
        'n_judges_cnt': data.n_judges,
        'n_pairs_cnt': ranking.n_pairs,
        'n_tiers_cnt': ranking.n_tiers,
        'n_abstentions_pairwise': ranking.n_abstentions,
        'n_items_with_ties_cnt': sum(count for count in ranking.ranks['rank'].value_counts() if count > 1),
        'n_items_in_top_tier_cnt': sum(ranking.ranks['rank'] == 1),
        'runtime': runtime_metrics.get('runtime', 0),
        'min_runtime': runtime_metrics.get('min_runtime', 0),
        'max_runtime': runtime_metrics.get('max_runtime', 0),
        'dissent_rate_dec': runtime_metrics.get('dissent_rate', np.nan),
    }
    return metrics

def process_spa_results(result, data, data_name):
    """Processes results specifically for the spa method."""
    if 'ranking_path' not in result:
        print(f"No ranking data in file.")
        return []

    ranking_path = result['ranking_path']
    dissent_rates = ranking_path.dissent_rates

    runtime_metrics = {
        'runtime': result.get('runtime', 0),
        'min_runtime': result.get('min_runtime', 0),
        'max_runtime': result.get('max_runtime', 0),
    }

    all_dissent_rate_results = []
    for dissent_rate in dissent_rates:
        print(f"Processing dissent rate: {dissent_rate}")
        ranking = ranking_path[dissent_rate]
        runtime_metrics['dissent_rate'] = dissent_rate

        judge_ranking_prefs = load_preferences(data.prefs, include_judge_id=True)
        standardized_ranking_prefs = load_preferences(ranking.get_prefs())
        metrics = calculate_ranking_metrics(data, ranking, runtime_metrics)
        metrics.update(calculate_comparison_metrics(judge_ranking_prefs, standardized_ranking_prefs))

        all_dissent_rate_results.extend(prepare_result_rows(data_name, 'spa', dissent_rate, metrics))

    return all_dissent_rate_results

def process_non_spa_results(result, data, data_name, method_name):
    """Processes results for non-spa methods."""

    ranking = result['ranking']
    judge_ranking_prefs = load_preferences(data.prefs, include_judge_id=True)
    standardized_ranking_prefs = load_preferences(ranking.get_prefs())

    runtime_metrics = {
        'runtime': result.get('runtime', 0),
        'min_runtime': result.get('min_runtime', 0),
        'max_runtime': result.get('max_runtime', 0),
        'dissent_rate': result.get('dissent_rate', np.nan),
    }

    metrics = calculate_ranking_metrics(data, ranking, runtime_metrics)
    metrics.update(calculate_comparison_metrics(judge_ranking_prefs, standardized_ranking_prefs))

    return prepare_result_rows(data_name, method_name, 'N/A', metrics)

def prepare_result_rows(data_name, method_name, dissent_rate, metrics):
    """Prepares structured result rows combining metadata with computed metrics."""
    result_rows = []
    for metric_name, metric_value in metrics.items():
        result_dict = {
            'data_name': data_name,
            'method_name': method_name,
            'sample_type': 'dissent',
            'metric_name': metric_name,
            'metric_value': metric_value,
            'dissent_rate': round(dissent_rate, 7) if dissent_rate != "N/A" else 'N/A'
        }
        result_rows.append(result_dict)
    return result_rows

def process_file(root, file, data_name, method_name):
    """Processes a single results file."""
    file_path = Path(root) / file
    try:
        with open(file_path, 'rb') as f:
            result = dill.load(f)

        data = PreferenceDataset.load(file=get_processed_data_file(data_name))

        if method_name == 'spa':
            return process_spa_results(result, data, data_name)
        else:
            return process_non_spa_results(result, data, data_name, method_name)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


output_dir = results_dir
all_results = []

for data_name in settings['data_names']:
    for method_name in settings['method_names']:
        file_pattern = f"{data_name}_{method_name}*.results"
        for root, _, files in os.walk(results_dir):
            for file in files:
                if (file.endswith('.results') and
                        fnmatch.fnmatch(file, file_pattern) and
                        "sampling" not in file and
                        "robustness" not in file and
                        "path" not in file and
                        "original" not in file and
                        "all" not in file and
                        "-1" not in file and
                        "_0" not in file):
                    print(f"Processing {data_name} with {method_name}")
                    all_results.extend(process_file(root, file, data_name, method_name))

base_results_df = pd.DataFrame(all_results)

if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

# Find the most recent existing results file
existing_results_files = sorted(output_dir.glob("base_results_*.results"), key=os.path.getmtime, reverse=True)

output_file = output_dir / f"base_results_{datetime.now().strftime('%Y-%m-%d')}_all.results"

if settings["modify_existing"] and existing_results_files:
    # Load the most recent existing results file
    with open(existing_results_files[0], 'rb') as f:
        existing_results_data = dill.load(f)
    existing_results_df = existing_results_data['results']

    # Ensure that both DataFrames have the same columns
    missing_in_existing = set(base_results_df.columns) - set(existing_results_df.columns)
    missing_in_base = set(existing_results_df.columns) - set(base_results_df.columns)

    for col in missing_in_existing:
        existing_results_df[col] = np.nan

    for col in missing_in_base:
        base_results_df[col] = np.nan

    # Sort the columns to ensure they are in the same order before updating
    existing_results_df = existing_results_df.sort_index(axis=1)
    base_results_df = base_results_df.sort_index(axis=1)

    # Identify rows to be updated
    keys = ["data_name", "method_name", "sample_type", "dissent_rate", "metric_name"]
    base_results_df_indexed = base_results_df.set_index(keys)
    existing_results_df_indexed = existing_results_df.set_index(keys)

    # Update existing rows with new data
    for index, row in base_results_df_indexed.iterrows():
        if index in existing_results_df_indexed.index:
            existing_results_df_indexed.loc[index] = row.values
        else:
            existing_results_df_indexed = pd.concat([existing_results_df_indexed, row.to_frame().T])

    # Convert back to DataFrame and reset column order
    updated_results_df = existing_results_df_indexed.reset_index()
    updated_results_df = updated_results_df.reindex(columns=["data_name", "method_name", "sample_type", "dissent_rate", "metric_name", "metric_value"])

    # Sort the index for better performance
    updated_results_df = updated_results_df.sort_index()

    # Save the updated results
    with open(output_file, 'wb') as f:
        dill.dump({'results': updated_results_df}, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

    print(f"Updated existing results file: {existing_results_files[0]}")

else:
    if settings["save_results"]:
        with open(output_file, 'wb') as f:
            dill.dump({'results': base_results_df}, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

        print(f"Saved new results to: {output_file}")