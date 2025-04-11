import os
from pathlib import Path
from argparse import ArgumentParser
import dill
from reporting.utils import make_report
from psutil import Process
import shutil

# Define directories
from sra.paths import reports_dir, templates_dir, repo_dir, get_results_file, overleaf_dir, reporting_dir

# Script settings / default command line arguments
settings = {
    "data_names": ["sushi_0.1", "sushi_0.2", "sushi_0.3", "sushi_0.4", "sushi_0.5", "sushi_0.6", "sushi_0.7", "sushi_0.8", "sushi_0.9", "sushi_1.0"],
    # "data_names": ["nba_coach_2021", "sushi", "survivor", "csrankings", "law","btl", "btl_modified"],
    "dissent_rate": None,
    "seed": 2338,
    "results_dir": "results",
    "report_template": "path_report.Rmd",
    "copy_to_overleaf": True,
    "make_report": True,  # Flag to control report generation
}

# Command-line argument parsing
if Process(pid=os.getppid()).name() not in ("pycharm"):
    p = ArgumentParser()
    p.add_argument('-d', '--data_names', type=str, nargs='+', default=settings['data_names'],
                    help='Names of the datasets (space-separated)')
    p.add_argument('-r', '--results_dir', type=str, default=settings['results_dir'],
                   help='Directory containing results files')
    p.add_argument('-t', '--report_template', type=str, default=settings['report_template'],
                   help='Report template file')
    p.add_argument('-o', '--output_dir', type=str, default=reports_dir, help='Directory to save the reports')
    p.add_argument('-M', '--merge_reports', action='store_true', help='Merge all reports into a single PDF')
    p.add_argument('-c', '--copy_to_overleaf', action='store_true', default=settings['copy_to_overleaf'],
                   help='Copy figures to Overleaf directory')
    p.add_argument('-m', '--make_report', action='store_true', default=settings['make_report'],
                   help='Generate reports using make_report')
    args, _ = p.parse_known_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    settings.update(args)


# Define directories
results_dir = Path(repo_dir) / settings["results_dir"]
report_template = templates_dir / settings["report_template"]
reports_dir.mkdir(parents=True, exist_ok=True)

for data_name in settings['data_names']:  # Loop through data_names
    method_name = "sra" # Method name is always sra
    print(f"Creating report for {data_name} using {method_name}")

    if settings['dissent_rate'] is not None:
        file_format = f"{data_name}_{method_name}_{settings['dissent_rate']}.results"
    else:
        file_format = f"{data_name}_{method_name}.results"

    print(f"Looking for files with format {file_format}")
    results_files = []
    for file_path in results_dir.iterdir():
        if file_path.is_file() and file_path.suffix == ".results" and str(file_format) in file_path.name:
            results_files.append(file_path)

    print(results_files)

    for results_file in results_files:
        print(f"Creating report for {results_file}")
        with open(results_file, 'rb') as f:
            results = dill.load(f)

        template_file = report_template
        output_file = reports_dir / f"{results_file.stem}_report.pdf"
        build_dir = reports_dir / results_file.stem

        if settings['make_report']:
            report_pdf = make_report(
                template_file=template_file,
                report_data_file=results_file,
                report_python_dir=repo_dir,
                output_file=output_file,
                build_dir=build_dir,
                clean=True,
                quiet=False,
                remove_build=False,
                remove_output=False,
            )

            print(f"Report created: {report_pdf}")

        # Copy figures to Overleaf directory if the flag is set
        if settings['copy_to_overleaf']:

            overleaf_figures_dir = overleaf_dir / "figures"
            print(f"Copying figures to {overleaf_figures_dir}")
            overleaf_figures_dir.mkdir(parents=True, exist_ok=True)
            figure_dir = reporting_dir / f"path_{data_name}_sra_report"

            if figure_dir.exists():
                print(f"Copying figures from {figure_dir}")
                for file in figure_dir.iterdir():
                    print(f"Checking {file}")
                    if file.is_file() and file.suffix == ".pdf":
                        print(f"Copying {file} to {overleaf_figures_dir}")
                        shutil.copy(file, overleaf_figures_dir)
            else:
                print(f"Figure directory not found: {figure_dir}")