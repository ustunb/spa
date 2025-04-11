"""
This script creates processed datasets for rank aggregation problems from CSV files.

We assign unique [method_name]s for specific rank aggregation problems.

We store the raw data used to specify each problem at:

    `tiered-rankings/data/[method_name]/`


1. [method_name]_prefs.csv: table of pairwise preferences on a collection of items and a panel of judges
2. [method_name]_judges.csv: table of judges (used to provide information about each judge)
3. [method_name]_items.csv: table of items (used to provide information about each item)
4. [method_name]_ranks.csv: table of rankings [method_name]_ranks.csv

2/3 are optional - we can use them to check the pairwise preferences expressed in 1 and to store metadata about judges/items (e.g., names)

4. is an alternative way to store pairwise preferences that might be convenient in certain situations. Sometimes it is easier to create a CSV of rankings from each judge rather than pairwise preferences (e.g., if judges have scored items or ranked all items). In such cases, we can store the rankings directly and then convert to pairwise preferences using the ranks_to_preferences function.
"""

import sys
import os
from psutil import Process
from argparse import ArgumentParser
from pathlib import Path

# Set working directory to parent directory
if '__file__' in globals():
    script_dir = Path(__file__).resolve().parent
    sys.path.append(str(script_dir.parent))

from spa.paths import get_processed_data_file, get_raw_data_file
from spa.data import PreferenceDataset

# script settings / default command line arguments
settings = {
    "data_names": ["synthetic100"],
    "seed": 2338,
}

# parse arguments when script is run from Terminal / not iPython console
if Process(pid=os.getppid()).name() not in ("pycharm", "python"):
    p = ArgumentParser()
    p.add_argument('-d', '--data_names', type=str, nargs='+', default=settings['data_names'], help='names of datasets')
    p.add_argument('-s', '--seed', type=int, default=settings['seed'], help='seed for random number generation')
    args, _ = p.parse_known_args()
    settings.update(vars(args))

if isinstance(settings['data_names'], str):
    settings['data_names'] = [settings['data_names']]

# Process each dataset in the provided list
for data_name in settings['data_names']:
    print(f"Processing data for {data_name}")

    data = PreferenceDataset.parse(file = get_raw_data_file(data_name))
    print("Raw data loaded")


    data.save(file = get_processed_data_file(data_name, check_save = True))
    print(f"Processed data saved for {data_name}")


