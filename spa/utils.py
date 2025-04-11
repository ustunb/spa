"""
this file contains general purpose helper functions
"""

_LOG_TIME_FORMAT = "%m/%d/%y @ %I:%M %p"


to_preference_value = lambda a, b: 1 if a < b else -1 if a > b else 0

def check_names(names):
    """
    Check that names are unique and non-empty
    :param names: List
    """
    assert isinstance(names, list), 'names must be a list'
    assert len(names) == len(set(names)), 'names must be unique'
    assert [len(str(name)) > 0 for name in names], 'names should be 1 character long'
    return True

def merge_pdfs(pdf_files, output_file):
    """
    Merge pdf files
    :param pdf_files: List of pdf files
    :param output_file: Output file
    """
    from PyPDF2 import PdfMerger
    merger = PdfMerger()
    for pdf in pdf_files:
        merger.append(pdf)
    merger.write(output_file)
    merger.close()



