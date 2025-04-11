library(tidyverse)
library(kableExtra)

#### Set directories and file paths ####
comp_name <- ""
output_dir <- if (comp_name == "kadekool") {
    "/Users/kadekool/Apps/Overleaf/selective-rank-aggregation/tables/"
} else {
    ""
}

#### Define file paths for output tables ####
kable_tex_file_none <- sprintf('%sbigtable.tex', output_dir)

#### Load data ####
raw_file <- "tiered-rankings/results/summary_statistics.csv"
raw_df <- read_csv(raw_file) %>%
    filter(data_name %in% c("survivor", "nba_coach_2021", "sushi", "csrankings"), sample_type == "none")

#### Prepare method data ####
# Convert metric_value to character to avoid type conflict during pivoting
method_df <- raw_df %>%
    filter(method_name != "sra") %>%
    mutate(metric_value = as.character(metric_value)) %>%  # Convert to character
    pivot_wider(
        names_from = method_name,  
        values_from = metric_value,  
        values_fill = list(metric_value = "--")  # Replace NA with '--' as a character
    )

#### Prepare SRA_0 data ####
# Filter for SRA with dissent_rate == 0, rename as SRA_0, and pivot
sra_df_0 <- raw_df %>%
    filter(method_name == "sra" & dissent_rate == 0) %>%
    mutate(metric_value = as.character(metric_value)) %>%  # Ensure all values are characters
    mutate(method_name = "SRA0") %>%
    pivot_wider(
        names_from = method_name,
        values_from = metric_value,
        values_fill = list(metric_value = "--")  # Ensure consistency with "--" as character
    )

#### Prepare SRA_1 data ####
# Get the number of items for each dataset
n_judges_df <- raw_df %>%
    filter(metric_name == "n_judges") %>%
    select(data_name, n_judges = metric_value) %>%
    mutate(n_judges = as.numeric(n_judges))  # Ensure n_judges is numeric

sra_df_1 <- map_dfr(n_judges_df$data_name, function(current_data_name) {
    # Get the number of items for the current dataset
    n_judges <- n_judges_df %>% filter(data_name == current_data_name) %>% pull(n_judges)
    
    # Calculate dissent rate as 1/n_judges
    dissent_rate_1 <- 1 / n_judges[1]  # Correct indexing
    
    
    # Filter SRA rows based on the calculated dissent_rate
    filtered_df <- raw_df %>%
        filter(method_name == "sra" & data_name == current_data_name & dissent_rate == dissent_rate_1)
    
    if (nrow(filtered_df) > 0) {
        # Pivot wider for SRA_1 if a matching row is found
        filtered_df %>%
            mutate(metric_value = as.character(metric_value), method_name = "SRA1") %>%
            pivot_wider(names_from = method_name, values_from = metric_value, values_fill = list(metric_value = "--"))
    } else {
        # Return an empty dataframe if no matching rows are found
        tibble(data_name = current_data_name, sample_type = NA, metric_name = NA, SRA_1 = "--")
    }
})

#### Prepare SRA_Max data ####
# Filter for the SRA row with the maximum dissent_rate (0.499), rename as SRA_Max, and pivot
sra_df_max <- raw_df %>%
    filter(method_name == "sra" & dissent_rate >= 0.499) %>%
    mutate(metric_value = as.character(metric_value)) %>%
    distinct(data_name, dissent_rate, sample_type, metric_name, .keep_all = TRUE) %>%  # Remove duplicates
    select(-sample_id) %>%  # Drop the sample_id column since it's not needed
    mutate(method_name = "SRAMax") %>%
    pivot_wider(
        names_from = method_name,
        values_from = metric_value
    ) %>%
    mutate(across(everything(), ~replace_na(., "--")))  # Replace NAs with '--'

#### Load sampling results ####
sampling_file <- "tiered-rankings/sampling_results.csv"
sampling_df <- read_csv(sampling_file)

#### Prepare sampling data (from sampling_results.csv) ####
#### Prepare sampling data (from sampling_results.csv) ####
# Pivot sampling results longer to match structure
sampling_df_long <- sampling_df %>%
    pivot_longer(
        cols = -c(data_name, sampling_type, method_name, dissent_rate), 
        names_to = "metric_name", 
        values_to = "metric_value"
    )

#### Pivot sampling results to ensure `metric_name` becomes columns ####
sampling_df_long_pivoted <- sampling_df_long %>%
    pivot_wider(
        names_from = method_name,  # Pivot by `method_name` to create new columns for each method
        values_from = metric_value  # The corresponding values for those methods
    ) %>%
    mutate(across(everything(), as.character)) %>%  # Ensure all columns are character types to match combined_df
    mutate(across(everything(), ~replace_na(.x, "--")))  # Replace NA values with '--'

#### Combine all datasets into one final dataframe ####
final_df <- bind_rows(
    combined_df, 
    sampling_df_long_pivoted %>%
        select(data_name, sampling_type, metric_name, kemeny, mc, copeland, borda, SRA_0, SRA_Max)  # Ensure consistency in method columns
)

#### Drop unnecessary columns ####
final_df <- final_df %>%
    select(-contains("dissent_rate"), -contains("sample_id"))

#### Add placeholder for SRA_1 if missing ####
final_df <- final_df %>%
    mutate(SRA_1 = ifelse(is.na(SRA_1), "--", SRA_1))

#### Set headers ####
sub_headers <- c("Dataset", "Sample Type", "Metric", "Kemeny", "MC", "Copeland", "Borda", "SRA_0", "SRA_1", "SRA_Max")

#### Create the LaTeX table ####
table_output <- final_df %>%
    kable(
        booktabs = TRUE,
        escape = FALSE,  # Escape LaTeX characters
        col.names = sub_headers,  # Use the correct column headers
        format = "latex",
        table.envir = NULL,
        linesep = ""  # Prevent the use of \addlinespace
    ) %>%
    add_header_above(
        c(" " = 3, "Methods" = 7), bold = FALSE, escape = FALSE  # Apply "Methods" only to columns 4-10
    ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down")) %>%  # Ensure the table doesn't overflow the page
    column_spec(1, latex_column_spec = "r", bold = TRUE) %>%  # Make the first column bold
    column_spec(4:10, width = "1.5cm") %>%  # Adjust the width of the method columns for better alignment
    row_spec(0, bold = TRUE) %>%  # Bold the header row
    row_spec(1:nrow(final_df), hline_after = TRUE)  # Add hline after every row for readability

#### Save the formatted table ####
cat(table_output, file = file.path("bigtable.tex"))