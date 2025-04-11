"""
This script loads a preference dataset, fits a ranking model using a specified method, and saves the results.

Usage:
    python dev_sra.py [-d DATA_NAME] [-r REWEIGHT] [-s SEED]

Arguments:
    -d, --data_name      : Name of the dataset (default: "survivor")
    -r, --reweight       : Flag to enable reweighting (default: False)
    -s, --seed           : Seed for random number generation (default: 2338)
"""

# generic imports
import os
import time
from psutil import Process
from argparse import ArgumentParser

# Settings dictionary for default configurations
settings = {
    "data_name": "synthetic100",
    "seed": 2338,
    "load_data": False,
    "create_report": False,
    "method_name": 'spa',
    'dissent': None,
    }

# Check if script is executed outside of PyCharm
if Process(pid=os.getppid()).name() not in ("pycharm"):
    p = ArgumentParser()
    p.add_argument('-d', '--data_name', type=str, default=settings['data_name'], help='name of dataset')
    p.add_argument('-s', '--seed', type=int, default=settings['seed'], help='random seed')
    p.add_argument('-r', '--create_report', type=bool, default=settings['create_report'], help='create report flag')
    p.add_argument('-l', '--load_data', type=bool, default=settings['load_data'], help='load data flag')
    args, _ = p.parse_known_args()
    settings.update(vars(args))

# Necessary imports
import dill
from spa.paths import get_processed_data_file, get_results_file, repo_dir, reports_dir, templates_dir, reporting_dir
from spa.data import PreferenceDataset
from spa.spa import SelectiveRankAggregator as Aggregator
import pandas as pd
pd.set_option('display.max_columns', None)


# Load dataset
data = PreferenceDataset.load(file=get_processed_data_file(**settings))

start_time = time.time()
fitter = Aggregator(
        data,
        exceed_max_dissent = True
        )

results = {
    **settings,
    'edge_weights': fitter.edge_weights
    }

if settings["dissent"] is None:

    fitter = Aggregator(
        data,
        exceed_max_dissent=True
    )

    edge_weights = fitter.edge_weights

    ranking_path, ranking_time = fitter.fit_path(prefs = data.prefs, upper_bound=0.5)


    results.update({
        'ranking_path': ranking_path,
        'runtime': time.time() - start_time,
        'min_runtime': min(ranking_time),
        'max_runtime': max(ranking_time),
        'total_runtime': sum(ranking_time),
        })
else:
    ranking, ranking_time = fitter.fit(items =data.items, ignore_missing = settings["ignore_missing"], dissent_rate = settings["dissent"])

results_file = get_results_file(**settings)
with open(results_file, 'wb') as f:
    dill.dump(results, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

if settings["create_report"]:
    print("Creating report")
    from reporting.utils import make_report
    output_file = reports_dir / f"path_{results_file.stem}_report.pdf"
    build_dir = reporting_dir / output_file.stem
    report_pdf = make_report(template_file=templates_dir / 'path_report.Rmd',
                             report_data_file=results_file,
                             report_python_dir=repo_dir,
                             output_file=output_file,
                             build_dir=build_dir,
                             clean=True,
                             quiet=False,
                             remove_build=False,
                             remove_output=False)