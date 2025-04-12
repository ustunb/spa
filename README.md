# SPA
Selective Preference Aggregation
```
├── data         # processed datasets                `data_dir`
├── spa          # source code                       `pkg_dir`
├── scripts      # scripts that run source code                            
├── results      # results                           `results_dir`
├── reporting    # source code for reporting
├── reports      # reports                           `reports_dir`          			
```

### 1. Add Your Dataset

1.  **Location:** Create a subfolder for your dataset within the `data/` directory (e.g., `data/movie/`).
2.  **File Naming:** Place your data file(s) in this subfolder, named according to the pattern: `{dataset_name}_{type}.csv` (e.g., `movie_ratings.csv`, `movie_pairwise.csv`).
3.  **Data Format:** Ensure your CSV file matches one of the following structures based on the `{type}` in your filename:

    * **For `ranking` or `rating` types:**
        * **Headers (First Row):** User ids.
        * **First Column:** Item ids.
        * **Cell A1 (Top-Left):** Must contain the exact text `item_name`.
        * **Data Cells:** Preference values (rating or rank) given by the user (column) for the item (row)

        **Example (`movie_ratings.csv`):**
        ```csv
        item_name,user101,user102,user103
        ItemA,5,4,3
        ItemB,3,4,5
        ItemC,2,1,4
        ItemD,1,2,2
        ```

    * **For `pairwise` comparison types:**
        * **Headers (First Row):** Must be exactly `judge_id`, `item_id_1`, `item_id_2`, `pref`.
        * **Rows:** Each row represents a single comparison made by a `judge_id`.
        * **`pref` Column:** Indicates the user preference:
            * `1`: `item_id_1` is preferred over `item_id_2`.
            * `-1`: `item_id_2` is preferred over `item_id_1`.
            * `0`: Represents a tie or indifference.

        **Example (`movie_pairwise.csv`):**
        ```csv
        judge_id,item_id_1,item_id_2,pref
        judgeA,itemX,itemY,1
        judgeA,itemY,itemZ,-1
        judgeB,itemX,itemY,0
        judgeB,itemX,itemZ,1
        judgeC,itemY,itemZ,1
        ```

### 2. Configure and Run Experiments

1.  **Update Dataset Creation Script:**
    * Open the file `scripts/create_datasets.py`.
    * Find the `settings` dictionary.
    * Add the `{dataset_name}` string (e.g., `"movie"`) from Step 1 to the `data_names` list.
        ```python
        settings = {
            "data_names": ["movie"], 
            ...
        }
        ```
    * Execute the script

2. **Update Main Experiment Script:**
    * Open the file `scripts/dev_spa.py`.
    * Find the `settings` dictionary within this script.
    * Add the same `{dataset_name}` string to the `data_names` list (or similar configuration entry) in this file as well.
        ```python
        settings = {
            "data_names": ["movie"], # Added "movie"
            "seed": 2338,
            ...
        }
        ```
    * Execute dev_spa.py

### 3. Generating and Viewing Results

1.  **Aggregate Base Results:**
    * Configure settings within  `scripts/aggregate_base_results.py`.
    * Execute the script .

2.  **Combine All Results:**
    * Configure settings within `scripts/combine_all_results.py`.
    * Execute the script.

3.  **Locate Output CSV:**
    * The final, combined results are saved as a CSV file within the `results/` directory (`results_dir`).
    * The filename includes a timestamp for uniqueness.

4.  **(Optional) Generate LaTeX Table:**
    * Modify and run the R script located at `scripts/create_big_table.R`. Update the script to point to the correct input CSV file from the previous step.
    * This will output a `.tex` file.

---
