import pathlib
import os
from datetime import datetime
from spa.paths import results_dir
import dill
import pandas as pd

output_dir = results_dir

all_results = []

for results_type in ["base_results"]:

    all_files = [f for f in results_dir.iterdir() if
                     f.is_file() and f.name.endswith(".results") and results_type in f.name]
    all_files = [f for f in all_files if "_all" in f.name]
    all_files = sorted(all_files, key=lambda f: os.path.getmtime(f), reverse=True)



    data = None
    if len(all_files) < 1:
        print(f"No files found for {results_type}")
        continue

    with open(all_files[0], 'rb') as f:
        data = dill.load(f)

    results = data['results']

    all_results.append(results)


all_results_df = pd.concat(all_results, ignore_index=True)

metric_value_map = {
    "inversions_median": "delta_inversions_med_pairwise",
    "inversions_min": "delta_inversions_min_pairwise",
    "inversions_max": "delta_inversions_max_pairwise",
    "abstentions_median": "delta_abstentions_med_pairwise",
    "abstentions_min": "delta_abstentions_min_pairwise",
    "abstentions_max": "delta_abstentions_max_pairwise",
    "preferences_median": "delta_preferences_med_pairwise",
    "preferences_min": "delta_preferences_min_pairwise",
    "preferences_max": "delta_preferences_max_pairwise",
    "specifications_median": "delta_specificity_med_pairwise",
    "specifications_min": "delta_specificity_min_pairwise",
    "specifications_max": "delta_specificity_max_pairwise",
}

all_results_df["metric_name"] = all_results_df["metric_name"].map(metric_value_map).fillna(all_results_df["metric_name"])



all_results_df.to_csv(output_dir / f"all_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)