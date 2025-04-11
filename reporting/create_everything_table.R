#### load packages ####
packages = c('reticulate', 'tidyverse', 'stringr', 'kableExtra')
for (pkg in packages) {
    library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
}
options(dplyr.width = Inf, dplyr.print_max = 1e9)
options(stringsAsFactors = FALSE)

#### set directories ####
comp_name = ""
if (comp_name == "kadrekool"){
    overleaf_dir = "/Users/kadekool/Dropbox/Apps/Overleaf/"
    repo_dir = "/Users/kadekool/Apps/Overleaf/tiered rankings/tables";
} else if (comp_name == "berkmac"){
    repo_dir = "/Users/berk/Dropbox (Harvard University)/repos/tiered-rankings/"
    overleaf_dir = "/Users/berk/Dropbox (Harvard University)/Apps/Overleaf/selective-rank-aggregation/"
} else {
    overleaf_dir = "/home/kadekool/"
    repo_dir = "/home/kadekool/tiered-rankings/"
}
input_dir = paste0(repo_dir, "results/")
output_dir = paste0(overleaf_dir, "")
kable_tex_file = sprintf('%severythingtable.tex', output_dir)

#### dashboard ####

all_data_names = c("nba_coach_2021", "survivor", "law_modified", "csrankings", "sushi", "dices350")
all_method_names = c("srauni", "sramin", "sramaj", "borda", "copeland", "kemeny", "mc")

# parameters that are fixed for each table

#### cell_metric_names ####
cell_metric_names = c(
    # Baseline metrics 
    'bsl_total_disagreements_pairwise',
    'bsl_med_disagreement_per_judge_pairwise',
    'bsl_n_abstentions_pairwise', 
    'bsl_n_tiers_cnt',
    'bsl_n_items_in_top_tier_cnt',
    'bsl_dissent_rate_dec',
    'mis_delta_inversions_med_pairwise',
    'adv_delta_inversions_max_pairwise',
    'mis_delta_specificity_med_pairwise',
    'adv_delta_specificity_max_pairwise',
    'mis_delta_abstentions_med_pairwise',
    'adv_delta_abstentions_max_pairwise'
    
    # 'bsl_missing_pairwise',
    # 'bsl_med_disagreement_per_judge_pairwise',
    # 'bsl_n_items_with_ties_cnt',
    # 'bsl_min_disagreement_per_judge_cnt',
    # 'bsl_max_disagreement_per_judge_cnt',
    # 'bsl_disagreement_per_judge_range',
    #'bsl_n_items_cnt',
    #'bsl_n_tiers_cnt',
    #'bsl_n_pairs_cnt',
    #'bsl_n_judges_cnt',
    #'bsl_n_abstentions_cnt',
    #'bsl_average_disagreement_per_judge_cnt',
    #'bsl_n_ties_cnt',
    
    
    # Sampling (Mis) metrics
    # 'mis_delta_preferences_min_cnt',
    # 'mis_delta_preferences_med_cnt',
    # 'mis_delta_preferences_max_cnt',
    # 'mis_delta_inversions_min_cnt',
    # 'mis_delta_inversions_med_cnt',
    # 'mis_delta_inversions_max_cnt',
    # 'mis_delta_abstentions_min_cnt',
    # 'mis_delta_abstentions_med_cnt',
    # 'mis_delta_specificity_med_cnt',
    # 'mis_delta_abstentions_range',
    # 'mis_delta_specifications_range',
    # 'mis_delta_specificity_max_cnt',
    # 'mis_delta_specificity_min_cnt',
    # 'mis_delta_abstentions_max_cnt',
    # 'mis_distance_all_mean_cnt',
    #'mis_distance_all_min_cnt',
    #'mis_distance_all_max_cnt',
    # 'mis_disagreement_all_cnt',
    # 'mis_largest_change_top_item_cnt',
    #'mis_largest_change_any_item_cnt',
    # 'mis_total_distance_all_items_cnt',
    #'mis_distance_top1_max_cnt',
    #'mis_distance_top1_mean_cnt',
    #'mis_distance_top1_min_cnt',
    #'mis_disagreement_top1_min_cnt',
    #'mis_disagreement_top1_cnt',
    #'mis_disagreement_top1_max_cnt',
    #'mis_distance_top5_mean_cnt',
    # 'mis_distance_top5_max_cnt',
    #'mis_distance_top5_min_cnt',
    #'mis_disagreement_top5_cnt',
    #'mis_disagreement_top5_max_cnt',
    #'mis_disagreement_top5_min_cnt',
    # 'mis_disagreement_per_judge_mean_cnt',
    #'mis_disagreement_per_judge_min_cnt',
    #'mis_disagreement_per_judge_max_cnt',
    
    # Sampling metrics
    # 'adv_delta_preferences_min_cnt',
    # 'adv_delta_preferences_med_cnt',
    # 'adv_delta_preferences_max_cnt',
    # 'adv_delta_inversions_min_cnt',
    # 'adv_delta_inversions_med_cnt',
    # 'adv_delta_abstentions_med_cnt',
    # 'adv_delta_specificity_med_cnt',
    # 'adv_delta_inversions_max_cnt',
    # 'adv_delta_abstentions_min_cnt',
    # 'adv_delta_abstentions_range',
    # 'adv_delta_specifications_range'
    # 'adv_delta_specificity_max_cnt',
    # 'adv_delta_specificity_min_cnt',
    # 'adv_delta_abstentions_max_cnt'
    # 'adv_distance_all_mean_cnt',
    # 'adv_distance_all_min_cnt',
    # 'adv_distance_all_max_cnt',
    # 'adv_disagreement_all_cnt',
    # 'adv_largest_change_top_item_cnt',
    #'adv_largest_change_any_item_cnt',
    # 'adv_total_distance_all_items_cnt',
    # 'adv_distance_top1_max_cnt',
    # 'adv_distance_top1_mean_cnt',
    # 'adv_distance_top1_min_cnt',
    # 'adv_disagreement_top1_min_cnt',
    # 'adv_disagreement_top1_cnt',
    # 'adv_disagreement_top1_max_cnt',
    # 'adv_distance_top5_mean_cnt',
    # 'adv_distance_top5_max_cnt',
    # 'adv_distance_top5_min_cnt',
    # 'adv_disagreement_top5_cnt',
    # 'adv_disagreement_top5_max_cnt',
    # 'adv_disagreement_top5_min_cnt',
    # 'adv_disagreement_per_judge_mean_cnt'
    # 'adv_disagreement_per_judge_min_cnt',
    # 'adv_disagreement_per_judge_max_cnt'
)

#### METHOD_TITLES ####
METHOD_TITLES = c(
    "srauni" = "\\\\srauni{}",
    "sramin" = "\\\\sramin{}",
    # "srafirstuni" = "\\\\srafirstuni{}",
    # "srasignificant" = "\\\\srasignificant{}",
    "sramaj" = "\\\\sramaj{}",
    "borda" = "\\\\borda{}",
    "copeland" = "\\\\copeland{}",
    "kemeny" = "\\\\kemeny{}",
    "mc" = "\\\\mc{}"
    
)

#### DATASET_TITLES ####
DATASET_TITLES = c(
    'nba_coach_2021' = '\\nbacoty{}',
    'survivor' = '\\survivordata{}',
    'csrankings' = '\\csrankingsdata{}',
    'sushi' = '\\sushidata{}',
    "btl" = "\\btldata{}",
    "btl_modified" = "\\btlmodifieddata{}", 
    'dices350' = "\\dicesdata{}",
    'law_modified' = "\\lawdata{}",
    "airlines" = "\\airlinesdata{}"
)

#### METRIC_TITLES ####
METRIC_TITLES = c(
    # Used Metrics
    'bsl_n_abstentions_pairwise' = "Abstention Rate",
    'bsl_total_disagreements_pairwise' = "Disagreement Rate",
    'bsl_n_tiers_cnt' = "\\# Tiers",
    'bsl_n_items_in_top_tier_cnt' = "\\# Top Items",
    'bsl_dissent_rate_dec' = "Dissent",
    'mis_delta_inversions_med_pairwise' = "$\\Delta$ Inversions- Sampling",
    'adv_delta_inversions_max_pairwise' = "$\\Delta$ Inversions -Gaming",
    'mis_delta_specificity_med_pairwise' = "$\\Delta$ Specifications -Sampling",
    'adv_delta_specificity_max_pairwise' = "$\\Delta$ Specifications -Gaming",
    'mis_delta_abstentions_med_pairwise' = "$\\Delta$ Abstentions -Sampling",
    'adv_delta_abstentions_max_pairwise' = "$\\Delta$ Abstentions -Gaming",
    
    # Baseline (Bsl) metrics (Unused)
    'bsl_n_items_with_ties_cnt' = "# Items with Ties",
    'bsl_min_disagreement_per_judge_cnt' = "Minimum Disagreement per User",
    'bsl_max_disagreement_per_judge_cnt' = "Maximum Disagreement per User",
    'bsl_med_disagreement_per_judge_pairwise' = "Median Disagreement per User",
    'bsl_n_items_cnt' = "\\# Items",
    'bsl_n_pairs_cnt' = "# Pairs",
    'bsl_n_judges_cnt' = "# Judges",
    'bsl_n_comparisons_cnt' = "# Comparisons",
    'bsl_average_disagreement_per_judge_cnt' = "Average Disagreement per user",
    'bsl_n_ties_cnt' = "# Ties",
    
    # Sampling (Mis) metrics (Unused)
    "mis_delta_specificity_med_pairwise" = "$\\Delta$ Median Specifications (Sampling)",
    'mis_delta_preferences_min_cnt' = "$\\Delta$ Preferences Min (Sampling)",
    'mis_delta_preferences_med_cnt' = "$\\Delta$ Median Sampling Preferences",
    'mis_delta_preferences_max_cnt' = "$\\Delta$ Preferences Max (Sampling)",
    'mis_delta_inversions_min_cnt' = "$\\Delta$ Inversions Min (Sampling)",
    'mis_delta_inversions_max_cnt' = "$\\Delta$ Inversions Max (Sampling)",
    'mis_delta_abstentions_min_cnt' = "$\\Delta$ Abstentions Min (Sampling)",
    'mis_delta_abstentions_med_cnt' = "$\\Delta$ Sampling Abstentions",
    'mis_delta_specificity_med_cnt' = "$\\Delta$ Specifications Med (Sampling)",
    'mis_delta_abstentions_range' = "$\\Delta$ Sampling Abstentions",
    'mis_delta_specifications_range' = "$\\Delta$ Specifications Range (Sampling)",
    'mis_delta_specificity_max_cnt' = "$\\Delta$ Specifications Max (Sampling)",
    'mis_delta_specificity_min_cnt' = "$\\Delta$ Specifications Min (Sampling)",
    'mis_delta_abstentions_max_cnt' = "$\\Delta$ Abstentions Max (Sampling)",
    'mis_distance_all_mean_cnt' = "Mean Distance (All Items, Sampling)",
    'mis_distance_all_min_cnt' = "Minimum Distance (All Items, Sampling)",
    'mis_distance_all_max_cnt' = "Maximum Distance (All Items, Sampling)",
    'mis_disagreement_all_cnt' = "Total Disagreements (Sampling)",
    'mis_largest_change_top_item_cnt' = "Largest $\\Delta$ (Top Item, Sampling)",
    'mis_largest_change_any_item_cnt' = "Largest $\\Delta$ (Any Item, Sampling)",
    'mis_total_distance_all_items_cnt' = "Total Distance (All Items, Sampling)",
    'mis_distance_top1_max_cnt' = "Maximum Distance (Top 1, Sampling)",
    'mis_distance_top1_mean_cnt' = "Mean Distance (Top 1, Sampling)",
    'mis_distance_top1_min_cnt' = "Minimum Distance (Top 1, Sampling)",
    'mis_disagreement_top1_min_cnt' = "Minimum Disagreement (Top 1, Sampling)",
    'mis_disagreement_top1_cnt' = "Disagreements (Top 1, Sampling)",
    'mis_disagreement_top1_max_cnt' = "Maximum Disagreement (Top 1, Sampling)",
    'mis_distance_top5_mean_cnt' = "Mean Distance (Top 5, Sampling)",
    'mis_distance_top5_max_cnt' = "Maximum Distance (Top 5, Sampling)",
    'mis_distance_top5_min_cnt' = "Minimum Distance (Top 5, Sampling)",
    'mis_disagreement_top5_cnt' = "Disagreements (Top 5, Sampling)",
    'mis_disagreement_top5_max_cnt' = "Maximum Disagreement (Top 5, Sampling)",
    'mis_disagreement_top5_min_cnt' = "Minimum Disagreement (Top 5, Sampling)",
    'mis_disagreement_per_judge_mean_cnt' = "Mean Disagreement per user (Sampling)",
    'mis_disagreement_per_judge_min_cnt' = "Minimum Disagreement per user (Sampling)",
    'mis_disagreement_per_judge_max_cnt' = "Maximum Disagreement per user (Sampling)",
    'mis_delta_preferences_range' = "$\\Delta$ Sampling Preferences",
    'mis_delta_inversions_range' = "$\\Delta$ Sampling Inversions",
    
    # Adversarial (Adv) metrics (Unused)

    'adv_delta_preferences_min_cnt' = "$\\Delta$ Preferences Min (Adversarial)",
    'adv_delta_preferences_med_cnt' = "$\\Delta$ Preferences Med (Adversarial)",
    'adv_delta_preferences_max_cnt' = "$\\Delta$ Preferences Max (Adversarial)",
    'adv_delta_inversions_min_cnt' = "$\\Delta$ Inversions Min (Adversarial)",
    'adv_delta_inversions_med_cnt' = "$\\Delta$ Inversions Med (Adversarial)",
    'adv_delta_abstentions_med_cnt' = "$\\Delta$ Abstentions Med (Adversarial)",
    'adv_delta_specificity_med_pairwise' = "$\\Delta$ Median Specifications  (Adversarial)",
    'adv_delta_inversions_med_pairwise' = "$\\Delta$ Median Inversions  (Adversarial)",
    'adv_delta_abstentions_min_cnt' = "$\\Delta$ Abstentions Min (Adversarial)",
    'adv_delta_abstentions_range' = "$\\Delta$ Adversarial Abstentions",
    'adv_delta_specifications_range' = "$\\Delta$ Specifications Range (Adversarial)",
    'adv_delta_specificity_max_cnt' = "$\\Delta$ Specifications Max (Adversarial)",
    'adv_delta_specificity_min_cnt' = "$\\Delta$ Specifications Min (Adversarial)",
    'adv_delta_abstentions_max_cnt' = "$\\Delta$ Adversarial Abstentions",
    'adv_distance_all_mean_cnt' = "Mean Distance (All Items, Adversarial)",
    'adv_distance_all_min_cnt' = "Minimum Distance (All Items, Adversarial)",
    'adv_distance_all_max_cnt' = "Maximum Distance (All Items, Adversarial)",
    'adv_disagreement_all_cnt' = "Total Disagreements (Adversarial)",
    'adv_largest_change_top_item_cnt' = "Largest $\\Delta$ (Top Item, Adversarial)",
    'adv_largest_change_any_item_cnt' = "Largest $\\Delta$ (Any Item, Adversarial)",
    'adv_total_distance_all_items_cnt' = "Total Distance (All Items, Adversarial)",
    'adv_distance_top1_max_cnt' = "Maximum Distance (Top 1, Adversarial)",
    'adv_distance_top1_mean_cnt' = "Mean Distance (Top 1, Adversarial)",
    'adv_distance_top1_min_cnt' = "Minimum Distance (Top 1, Adversarial)",
    'adv_disagreement_top1_min_cnt' = "Minimum Disagreement (Top 1, Adversarial)",
    'adv_disagreement_top1_cnt' = "Disagreements (Top 1, Adversarial)",
    'adv_disagreement_top1_max_cnt' = "Maximum Disagreement (Top 1, Adversarial)",
    'adv_distance_top5_mean_cnt' = "Mean Distance (Top 5, Adversarial)",
    'adv_distance_top5_max_cnt' = "Maximum Distance (Top 5, Adversarial)",
    'adv_distance_top5_min_cnt' = "Minimum Distance (Top 5, Adversarial)",
    'adv_disagreement_top5_cnt' = "Disagreements (Top 5, Adversarial)",
    'adv_disagreement_top5_max_cnt' = "Maximum Disagreement (Top 5, Adversarial)",
    'adv_disagreement_top5_min_cnt' = "Minimum Disagreement (Top 5, Adversarial)",
    'adv_disagreement_per_judge_mean_cnt' = "Mean Disagreement per user (Adversarial)",
    'adv_disagreement_per_judge_min_cnt' = "Minimum Disagreement per user (Adversarial)",
    'adv_disagreement_per_judge_max_cnt' = "Maximum Disagreement per user (Adversarial)",
    'adv_delta_preferences_range' = "$\\Delta$ Adversarial Preferences",
    'adv_delta_inversions_range' = "$\\Delta$ Adversarial Inversions"
)

#### table creation code ####

raw_df <- dir(input_dir, pattern = "all_results", full.names = TRUE) %>%
    file.info() %>%
    arrange(desc(mtime)) %>%
    rownames_to_column(var = "file") %>%
    pull(file) %>%
    first() %>%
    read.csv()

raw_df <- raw_df %>%
    mutate(across(where(is.character), ~str_replace_all(., "-", "_"))) %>%
    mutate(
        dissent_rate = na_if(dissent_rate, "N/A"), 
        dissent_rate = as.numeric(dissent_rate),
        dissent_rate = round(dissent_rate, 7),
    )


raw_df = raw_df %>%
    mutate(
        sample_type = case_when(
            sample_type == "robustness" ~ "adv",
            sample_type == "sampling" ~ "mis",
            .default = "bsl"
        )
    )

dissent_analysis <- raw_df %>%
    filter(method_name == "sra", metric_name %in% c('n_abstentions_cnt')) %>%  # Fixed parentheses for %in%
    group_by(data_name, method_name, sample_type) %>%
    arrange(dissent_rate) %>%
    summarize(
        base_dissent = first(metric_value[as.numeric(dissent_rate) == 0]),
        # dissent_first_uni = first(dissent_rate[as.numeric(dissent_rate) > 0 & metric_value != base_dissent], default = 0), # Default value
        # .groups = 'drop'
    ) %>%
    select(-sample_type, -base_dissent) %>%
    ungroup()

# dissent_significant <- raw_df %>%
#     filter(
#         method_name == "sra",
#         metric_name == "stable_sum",
#         as.numeric(metric_value) >= 95,
#         sample_type == "adv"
#     ) %>%
#     group_by(data_name) %>%
#     summarize(
#         dissent_significant = max(as.numeric(dissent_rate)),
#         .groups = 'drop'
#     )

dissent_df <- raw_df %>%
    filter(
        method_name == "sra",
        metric_name == "n_judges_cnt",
        dissent_rate != "0.0",      
        dissent_rate != "0",  
        !is.na(dissent_rate)         
    ) %>%
    mutate(dissent_rate = as.numeric(dissent_rate)) %>%
    filter(as.numeric(dissent_rate) > 0.0) %>%
    filter(as.numeric(dissent_rate) < 0.5) %>%
    select(data_name, method_name, dissent_rate, n_judges = metric_value) %>%
    mutate(
        dissent_rate = as.numeric(dissent_rate) * as.numeric(n_judges)
    ) %>%
    group_by(data_name, n_judges) %>%
    summarize(
        dissent_maj = max(dissent_rate),
        dissent_min = min(dissent_rate),
        dissent_uni = 0.0,
        .groups = 'drop'
    ) %>%
    ungroup() %>%
    mutate(
        dissent_maj = dissent_maj / n_judges,
        dissent_min = dissent_min / n_judges,
        dissent_uni = dissent_uni / n_judges
    ) 
# %>%
# left_join(dissent_analysis, by = "data_name") %>%
# left_join(dissent_significant, by = "data_name") %>%
# mutate(
#     dissent_first_uni = dissent_first_uni 
# )


processed_sra_df <- raw_df %>%
    filter(method_name == "sra") %>%
    left_join(dissent_df, by = c("data_name")) %>%
    mutate(dissent_rate = as.numeric(dissent_rate)) %>%
    crossing(dissent_type = c("dissent_uni", "dissent_min", "dissent_maj")) %>%
    mutate(
        dissent_value = case_when(
            dissent_type == "dissent_uni" ~ dissent_uni,
            dissent_type == "dissent_min" ~ dissent_min,
            dissent_type == "dissent_maj" ~ dissent_maj,
            TRUE ~ NA_real_
        ),
        method_name = case_when(
            dissent_type == "dissent_uni" ~ "srauni",
            dissent_type == "dissent_min" ~ "sramin",
            dissent_type == "dissent_maj" ~ "sramaj",
            TRUE ~ method_name
        )
    ) %>%
    filter(near(dissent_rate, dissent_value, tol = 1e-6)) %>%
    ungroup()

raw_results_df <- bind_rows(
    processed_sra_df,
    raw_df %>% filter(method_name != "sra")
) %>%
    select(-n_judges, -dissent_type, -dissent_value) %>%
    unite("metric_name", c("sample_type", "metric_name"), sep = "_") %>%
    arrange(data_name, method_name, metric_name)


table_results_df = raw_results_df %>%
    select(data_name, method_name, stat_value = metric_value, stat_name = metric_name) %>%
    group_by(data_name) %>%
    mutate(
        n_items = as.numeric(first(stat_value[stat_name == 'bsl_n_items_cnt'], default = NA)),
        n_pairs = as.numeric(first(stat_value[stat_name == 'bsl_n_pairs_cnt'], default = NA)),
        n_judges = as.numeric(first(stat_value[stat_name == 'bsl_n_judges_cnt'], default = NA)),
    ) %>%
    ungroup() %>%
    mutate(
        svalue = case_when(
            str_ends(stat_name, "_cnt") ~ sprintf("%1.0f", stat_value),
            str_ends(stat_name, "_dec") ~ sprintf("%1.4f", stat_value),
            stat_name == "bsl_total_disagreements_pairwise" ~ paste0(sprintf("%.1f", (as.numeric(stat_value) / (n_pairs * n_judges)) * 100), "\\%"),
            str_ends(stat_name, "_pairwise") ~ paste0(sprintf("%.1f", (as.numeric(stat_value) / n_pairs) * 100), "\\%"),
            TRUE ~ paste0(sprintf("%.1f", (as.numeric(stat_value)) * 100), "\\%")
        ),
        svalue = ifelse(is.na(stat_value) & stat_name == 'bsl_dissent_rate_dec', "--", svalue),
        svalue = ifelse(is.na(stat_value) & stat_name != 'bsl_dissent_rate_dec', "", svalue)
    ) %>%
    select(-stat_value) %>%
    distinct()

cells_df = table_results_df %>%
    filter(stat_name %in% cell_metric_names) %>%
    arrange(data_name, method_name, stat_name, desc(svalue)) %>%
    group_by(data_name, method_name, stat_name) %>%
    slice_tail(n = 1) %>%  # Keep only the last matching value for each group
    ungroup() %>%
    pivot_wider(
        names_from = stat_name,
        values_from = svalue
    ) %>%
    relocate(all_of(cell_metric_names), .after = last_col())


cells_df = cells_df %>%
    group_by(data_name, method_name) %>%
    unite(cell_str, sep = "\\\\", all_of(cell_metric_names)) %>%
    mutate(cell_str = sprintf("\\cell{r}{%s}\n", cell_str)) %>%
    ungroup()


#### metrics big table ####
table_df = cells_df %>%
    filter(data_name %in% all_data_names,
           method_name %in% all_method_names) %>%
    arrange(
        match(data_name, all_data_names),
        match(method_name, all_method_names),
    )

# create headers manually to avoid unique names issues
headers_df = table_df %>%
    mutate(method_name = str_to_lower(method_name)) %>%
    group_by(method_name) %>%
    distinct(method_name)

# bottom level columns
headers = headers_df %>%
    mutate(method_name = str_replace_all(method_name, METHOD_TITLES)) %>%
    pull(method_name) %>%
    prepend(c("Dataset", "Metrics"))

kable_df = table_df %>%
    mutate(
        metrics = "\\metricsguide{}",
        data_name = recode(data_name, !!!DATASET_TITLES),
        method_name = recode(method_name, !!!METHOD_TITLES),
    ) %>%
    pivot_wider(
        names_from = c(method_name),
        values_from = cell_str,
        names_sort = FALSE
    )

# # Explicitly reorder the columns based on `all_method_names`
kable_df = kable_df %>%
    select(c("data_name", "metrics", all_of(METHOD_TITLES)))


overview_table =  kable_df %>%
    kable(
        booktabs = TRUE,
        escape = FALSE,
        col.names = headers,
        format = "latex",
        table.envir = NULL,
        linesep = ""
    ) %>%
    #kable_styling(latex_options = c("repeat_header", "scale_down", latex_table_env = NULL)) %>%
    column_spec(column = 1, latex_column_spec = "l") %>%
    column_spec(column = 2, latex_column_spec = "r") %>%
    row_spec(2:nrow(kable_df)-1, hline_after = TRUE, extra_latex_after = "\n")

cell_metric_titles = METRIC_TITLES[cell_metric_names]
metrics_cmd = paste0("\\renewcommand{\\metricsguide}[0]{\\cell{r}{",paste0(cell_metric_titles, collapse = "\\\\"), "}}")
cat(metrics_cmd, file = kable_tex_file)
cat(overview_table, file = kable_tex_file, append=TRUE)