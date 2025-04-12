"""
This file demonstrates how to use the reporting tools
The file will compile the DemoReport.Rmd template
"""
from reporting.utils import open_file, make_report
from spa.paths import repo_dir, results_dir, reports_dir, templates_dir
import dill
import pandas as pd

# create toy data for a report
report_data = {
    'method_name': "iris",
    'df': pd.read_csv(
        'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'),
    }

# save toy data to disk
report_data_file = results_dir / 'iris.results'
with open(report_data_file, 'wb') as outfile:
    dill.dump(report_data, outfile, recurse = True)

# check that you can load the data you shaved
with open(report_data_file, 'rb') as infile:
    results = dill.load(infile)

# create a report using the 'demo_report.Rmd' template and 'report_data'
template_file = templates_dir / 'demo_report.Rmd'
output_file = reports_dir / 'iris_report.pdf'
build_dir = reports_dir / output_file.stem

report_pdf = make_report(template_file = template_file,
                         report_data_file = report_data_file,
                         report_python_dir = repo_dir,
                         output_file = output_file,
                         build_dir = build_dir,
                         clean = True,
                         quiet = False,
                         remove_build = False,
                         remove_output = False)

# open the report
open_file(report_pdf)

# build_dir contains source files for the report like tex files, figures etc.
for f in build_dir.iterdir():
    print(f)