import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(4338)

try:
    from sra.paths import data_dir
except ImportError:
    print("Warning: Could not import data_dir from sra.paths. Using './data'.")
    data_dir = Path('./data')

input_file = data_dir / 'csrankings' / 'csrankings_rankings.csv'
output_base_dir = data_dir

if not input_file.is_file():
    raise FileNotFoundError(f"Input file not found: {input_file}")

df = pd.read_csv(input_file)
print(f"Loaded data from {input_file}")

kept_pct_list = [0.5, 0.55 ,0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

for pct_kept in kept_pct_list:
    percentage = int(pct_kept * 100)
    print(f"Processing for {percentage}% target keep rate...")

    # modified_df = df.copy()
    #
    # # Modify all rows, but leave the first column (assumed to be csrankings names) unchanged.
    # target_row_indices = slice(None)
    # target_col_indices = slice(1, None)
    #
    # target_rows_count = modified_df.shape[0]
    # target_cols_count = modified_df.shape[1] - 1
    #
    # if target_rows_count > 0 and target_cols_count > 0:
    #     random_matrix = np.random.rand(target_rows_count, target_cols_count)
    #     mask_to_nan = random_matrix > pct_kept
    #
    #     target_data = modified_df.iloc[target_row_indices, target_col_indices]
    #     mask_df = pd.DataFrame(mask_to_nan, index=target_data.index, columns=target_data.columns)
    #     modified_data = target_data.where(~mask_df, np.nan)
    #
    #     modified_df.iloc[target_row_indices, target_col_indices] = modified_data
    #     print(f"  Applied probabilistic NaN replacement (target keep rate: {pct_kept*100:.1f}%).")
    # else:
    #     print("  DataFrame too small or no eligible cells to modify.")
    #
    # if modified_df.columns[0] != df.columns[0]:
    #      raise ValueError("First column name changed unexpectedly!")

    folder_dir = output_base_dir / f'csrankings_{percentage}'
    folder_dir.mkdir(parents=True, exist_ok=True)

    output_file = folder_dir / f'csrankings_{percentage}_rankings.csv'
    df.to_csv(output_file, index=False)
    print(f"  Saved modified data to {output_file}")

print("\nFinished processing and saving all modified csrankings ranking data.")
