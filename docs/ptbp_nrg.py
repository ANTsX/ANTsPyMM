import glob
import os
import pandas as pd
import shutil
from pathlib import Path
import antspymm

def create_directory_and_copy_file(file_path, destination_path):
    """
    Create the directory for the destination path and copy the file to the new location.

    Parameters:
    - file_path (str): The full path of the source file to be copied.
    - destination_path (str): The full path of the destination file.
    """
    destination_dir = Path(destination_path).parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(file_path, destination_path)

def create_directory_and_copy_similar_files(file_path, destination_path):
    """
    Create the directory for the destination path and copy the file and any similarly named files
    in the same directory to the new location.

    Parameters:
    - file_path (str): The full path of the source file to be copied.
    - destination_path (str): The full path of the destination file.
    """
    source_dir = Path(file_path).parent
    destination_dir = Path(destination_path).parent
    file_stem = Path(file_path).stem

    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(file_path, destination_path)

    for file in source_dir.glob(file_stem + '.*'):
        if file.name != Path(file_path).name:
            shutil.copy(file, destination_dir / file.name)

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

def process_files(pattern, modality, identifier):
    """
    Process files based on a glob pattern and a specific modality.

    Parameters:
    - pattern (str): Glob pattern to match files.
    - modality (str): Modality to process.
    - identifier (str): Identifier for file naming.
    """
    suids = glob.glob(pattern)
    if suids:
        for suid in suids:
            df = peds2nrg(suid, modality)
            print(df)
            mynrg = antspymm.nrg_format_path('PTBP', df['id'].iloc[0], df['dt'].iloc[0], 
                                             df['modality'].iloc[0], identifier, separator='-')
            if modality == 'DTI':
                create_directory_and_copy_similar_files(df['filename'].iloc[0], mynrg+'.nii.gz')
            else:
                create_directory_and_copy_file(df['filename'].iloc[0], mynrg+'.nii.gz')

# Process different modalities
rr = "images/PEDS*/*/*/"
process_files(rr+"*t1.nii.gz", 'T1w', '000')
process_files(rr+"*bold*.nii.gz", 'rsfMRI', '000')
process_files(rr+"*pcasl*.nii.gz", 'perf', '000')
# DTI specific processing
dti_nums = ['0011', '0014', '0017', '0020']
for num in dti_nums:
    process_files(f"images/PEDS*/*/*/*{num}_DTI*.nii.gz", 'DTI', num)
