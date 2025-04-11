"""
This file defines paths for key directories and files.
Contents include:

1. Directory Names: Paths of directories where we store code, data, results, etc.
2. File Name Generators: functions to name processed datasets, results, plots, etc.
"""

from pathlib import Path
import numpy as np

repo_dir = Path(__file__).resolve().parent.parent # path to the GitHub repository
pkg_dir = repo_dir / "spa/" # path to source code
test_dir = repo_dir / 'tests' # path to unit tests
data_dir = repo_dir / "data/" # path to datasets
results_dir = repo_dir / 'results' # path to results files
reporting_dir = repo_dir / 'reporting' # path to source code for reporting
templates_dir = reporting_dir / 'templates' # path we store reporting templates
reports_dir = repo_dir / 'reports' # path to reports'
scripts_dir = repo_dir / 'scripts' # path to scripts

# create local directories if they do not exist
results_dir.mkdir(exist_ok = True)
reports_dir.mkdir(exist_ok = True)

# Naming Functions
def get_processed_data_file(data_name, **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
    :return: Path of the processed data file on disk
    """
    #if method_name is a list/np.array, extract the first element
    if isinstance(data_name, (list, np.ndarray)):
        data_name = str(data_name[0])
    # assert isinstance(method_name, str) and len(method_name) > 0
    f = data_dir / f'{data_name}_processed.pickle'
    return f

def get_raw_data_file(data_name, **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
    :return: dictionary of all possible CSV files
    """
    #assert isinstance(method_name, str) and len(method_name) > 0
    preference_type = kwargs.get('preference_type')
    if preference_type is None:
        for pt in ['pairwise', 'ratings', 'rankings']:
            f = data_dir / data_name / f"{data_name}_{pt}.csv"
            if f.exists():
                preference_type = pt
                break
        assert preference_type is not None, "data type not found for {method_name}"
    f = data_dir / data_name / f"{data_name}_{preference_type}.csv"
    return f

def get_raw_data_component_files(data_name, **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
    :return: dictionary of all possible CSV files
    """
    #assert isinstance(method_name, str) and len(method_name) > 0
    templates = {
        'prefs': '{}_prefs.csv',
        'judges': '{}_judges.csv',
        'items': '{}_items.csv',
        'ranks': '{}_ranks.csv'
    }
    files = {k: data_dir / data_name / v.format(data_name) for k, v in templates.items()}
    return files

def get_results_file(data_name, method_name, **kwargs):
    """
    returns file name for pickle files used to store the results of a training job (e.g., in `train_classifier`)

    :param data_name: string containing name of the dataset
    :param method_name: string containing name of the classification method
    :param kwargs: used to catch other args when unpacking dictionaies
           this allows us to call this function as get_results_file_name(**settings)

    :return: Path of results object
    """
    #assert isinstance(method_name, str) and len(method_name) > 0
    # assert isinstance(method_name, str) and len(method_name) > 0

    if method_name == 'sra':
        dissent = kwargs.get('dissent_rate', float('nan'))
        if np.isfinite(dissent):
            dissent_str = f"{int(dissent * 100):03d}"
            f = results_dir / f'{data_name}_{method_name}_{dissent_str}.results'
            return f

    f = results_dir / f'{data_name}_{method_name}.results'

    return f

def get_all_sampling_results_files(data_name, method_name, sampling_type = 'sampling', **kwargs):
    """
    returns list of file names for pickle files used to store the results of a sampling job (e.g., in `sample_dataset`)
    """
    assert isinstance(sampling_type, str) and len(sampling_type) > 0

    out = f"{data_name}_{method_name}_{sampling_type}.results"
    # append dissent if we ran SRA for a specific dissent rate
    out = results_dir / out
    return out


def get_sampling_results_file(data_name, method_name, sampling_type = 'sampling', sample_id = 1, **kwargs):
    """
    returns file name for pickle files used to store the results of a sampling job (e.g., in `sample_dataset`)
    """
    #assert isinstance(method_name, str) and len(method_name) > 0
    assert isinstance(method_name, str) and len(method_name) > 0
    assert isinstance(sampling_type, str) and len(sampling_type) > 0
    assert isinstance(sample_id, int) and sample_id >= 0

    header = f"{data_name}_{method_name}"
    # append dissent if we ran SRA for a specific dissent rate
    dissent = kwargs.get('dissent_rate', float('nan'))
    if method_name == 'sra' and np.isfinite(dissent):
        header = f"{header}_dissent_{int(dissent * 100):03d}"

    out = f"{header}_{sampling_type}_{sample_id:03d}.results"
    #use pathlib to create the file path
    out = results_dir / out
    return out

