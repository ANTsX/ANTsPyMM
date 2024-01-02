import glob
import os
import pandas as pd
from pathlib import Path
import antspymm
import shutil
import re

def get_full_extension(filename):
    """
    Get the full extension from a filename that may contain multiple dots.

    Parameters:
    - filename (str): The filename to extract the extension from.

    Returns:
    - str: The full extension of the file, including all parts after the first dot.
    """
    # Find the first occurrence of a dot
    first_dot_index = filename.find('.')
    if first_dot_index != -1:
        full_extension = filename[first_dot_index:]
    else:
        full_extension = ''

    return full_extension

def create_directory_and_process_file(file_path, destination_path, use_symlinks=False):
    """
    Create the directory for the destination path and either copy the file or create a symbolic link
    to the file in the new location, based on the user's choice.

    Parameters:
    - file_path (str): The full path of the source file.
    - destination_path (str): The full path of the destination file or symbolic link.
    - use_symlinks (bool): If True, create symbolic links; otherwise, copy files.
    """
    destination_path = Path(destination_path)
    destination_dir = destination_path.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_path = os.path.abspath( file_path )
    if use_symlinks and not destination_path.exists():
        os.symlink(file_path, destination_path)
    elif not use_symlinks:
        shutil.copy(file_path, destination_path)

def create_directory_and_process_similar_files(file_path, destination_path, use_symlinks=False):
    """
    Create the directory for the destination path and either copy the file and any similarly named files
    or create symbolic links for them in the new location, based on the user's choice.

    Parameters:
    - file_path (str): The full path of the source file.
    - destination_path (str): The full path of the destination file or symbolic link.
    - use_symlinks (bool): If True, create symbolic links; otherwise, copy files.
    """
    file_path = os.path.abspath( file_path )
    file_path = Path(file_path)
    destination_path = Path(destination_path)
    source_dir = file_path.parent
    destination_dir = destination_path.parent
    file_stem = file_path.stem
    file_stem = Path(file_stem).stem

    destination_dir.mkdir(parents=True, exist_ok=True)
    if use_symlinks and not destination_path.exists():
        os.symlink(file_path, destination_path)
    elif not use_symlinks:
        shutil.copy(file_path, destination_path)

    for file in source_dir.glob(file_stem + '.*'):
#        destination_file = destination_dir / file.name
        newext = get_full_extension( str(file) )
        oldext = get_full_extension( str(destination_path) )
        newfilename = re.sub( oldext, newext, str(destination_path) )
        destination_file = Path(newfilename)
        if file.name != file_path.name:
            print( oldext )
            print( newext )
            print( newfilename )
            derka
            if use_symlinks and not destination_file.exists():
                os.symlink(file, destination_file)
            elif not use_symlinks:
                shutil.copy(file, destination_file)


def peds2nrg(file_path, outmod):
    """
    Reorganize file path information into a DataFrame.

    Parameters:
    - file_path (str): The file path to process.
    - outmod (str): The modality to be output.

    Returns:
    - pandas.DataFrame: A DataFrame containing the processed file path information.
    """
    temp = file_path.split(os.sep)[1:]
    mydf = pd.DataFrame({
        'id': [temp[0]],
        'dt': [temp[1]],
        'modality': [outmod],
        'imageID': ['000'],
        'filename': [file_path]
    })
    return mydf

def process_files(pattern, modality, identifier, use_symlinks=False):
    """
    Process files based on a glob pattern and a specific modality.

    Parameters:
    - pattern (str): Glob pattern to match files.
    - modality (str): Modality to process.
    - identifier (str): Identifier for file naming.
    - use_symlinks (bool): If True, create symbolic links; otherwise, copy files.
    """
    suids = glob.glob(pattern)
    if suids:
        for suid in suids:
            df = peds2nrg(suid, modality)
            print(df)
            mynrg = antspymm.nrg_format_path('PTBP', df['id'].iloc[0], df['dt'].iloc[0], 
                                             df['modality'].iloc[0], identifier, separator='-')
            destination_path = mynrg+'.nii.gz'
            if modality == 'DTI':
                create_directory_and_process_similar_files(df['filename'].iloc[0], destination_path, use_symlinks)
            else:
                create_directory_and_process_file(df['filename'].iloc[0], destination_path, use_symlinks)

# Example usage
rr = "data/PEDS*/*/*/"
use_symlinks = True  # Set to False to copy files instead
process_files(rr+"*t1.nii.gz", 'T1w', '000', use_symlinks)
process_files(rr+"*bold_fc_1.nii.gz", 'rsfMRI', '000', use_symlinks)
process_files(rr+"*bold_fc_2.nii.gz", 'rsfMRI', '001', use_symlinks)
process_files(rr+"*pcasl*.nii.gz", 'perf', '000', use_symlinks)

# DTI specific processing
dti_nums = [f"{i:04d}" for i in range(25)]
for num in dti_nums:
    process_files(rr +"*"+num+"_DTI*.nii.gz", 'DTI', num, use_symlinks)
