import os
from pathlib import Path
from argparse import ArgumentParser
import dill
from reporting.utils import make_report
from psutil import Process
# Define directories
from sra.paths import reports_dir, templates_dir, repo_dir, get_results_file

# Script settings / default command line arguments
settings = {
    "data_names": ["dices350"],
    "method_names": ["kemeny", "borda", "copeland", "mc"],
    "dissent_rate": None,
    "seed": 2338,
    "results_dir": "results",
    "report_template": "demo_report.Rmd",
}

# Command-line argument parsing
if Process(pid=os.getppid()).name() not in ("pycharm"):
    p = ArgumentParser()
    p.add_argument('-d', '--data_names', type=str, nargs='+', default=settings['data_names'],
                    help='Names of the datasets (space-separated)')
    p.add_argument('-m', '--method_names', type=str, nargs='+', default=settings['method_names'],
                    help='Names of the methods (space-separated)')
    p.add_argument('-r', '--results_dir', type=str, default=settings['results_dir'],
                   help='Directory containing results files')
    p.add_argument('-t', '--report_template', type=str, default=settings['report_template'],
                   help='Report template file')
    p.add_argument('-o', '--output_dir', type=str, default=reports_dir, help='Directory to save the reports')
    p.add_argument('-M', '--merge_reports', action='store_true', help='Merge all reports into a single PDF')
    args, _ = p.parse_known_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    settings.update(args)

# Define directories
results_dir = Path(repo_dir) / settings["results_dir"]
report_template = templates_dir / settings["report_template"]
reports_dir.mkdir(parents=True, exist_ok=True)

for data_name in settings['data_names']:
    for method_name in settings['method_names']:

        file_format = get_results_file(data_name, method_name)
        if settings['dissent_rate'] is not None:
            file_format = f"{data_name}_{method_name}_{settings['dissent_rate']}.results"

        results_files = []
        for file_path in results_dir.iterdir():
            if file_path.is_file() and file_path.suffix == ".results" and file_format in file_path.name:
                results_files.append(file_path)

        for results_file in results_files:
            print(f"Creating report for {results_file}")
            with open(results_file, 'rb') as f:
                results = dill.load(f)

            template_file = report_template
            output_file = reports_dir / f"{results_file.stem}_report.pdf"
            build_dir = reports_dir / results_file.stem
            report_pdf = make_report(
                template_file=template_file,
                report_data_file=results_file,
                report_python_dir=repo_dir,
                output_file=output_file,
                build_dir=build_dir,
                clean=False,
                quiet=False,
                remove_build=False,
                remove_output=False,
            )
            print(f"Report created: {report_pdf}")
