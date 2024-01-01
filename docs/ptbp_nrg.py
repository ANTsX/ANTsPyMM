import glob
import os
import pandas as pd
import shutil
from pathlib import Path
import antspymm

import os
import shutil
from pathlib import Path

def create_directory_and_copy_similar_files(file_path, destination_path):
    """
    Create the directory for the destination path and copy the file and any similarly named files
    in the same directory to the new location.

    Parameters:
    file_path (str): The full path of the source file to be copied.
    destination_path (str): The full path of the destination file.
    """
    source_dir = Path(file_path).parent
    destination_dir = Path(destination_path).parent
    file_stem = Path(file_path).stem
    file_stem = Path(file_stem).stem

    # Create destination directory if it doesn't exist
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Copy the specified file
    shutil.copy(file_path, destination_path)

    # Find and copy similarly named files
    for file in source_dir.glob(file_stem + '.*'):
        if file.name != Path(file_path).name:
            shutil.copy(file, destination_dir / file.name)

def create_directory_and_copy_file(file_path, destination_path):
    """
    Create the directory for the destination path and copy the file to the new location.

    Parameters:
    file_path (str): The full path of the source file to be copied.
    destination_path (str): The full path of the destination file.
    """
    destination_dir = Path(destination_path).parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(file_path, destination_path)
def peds2nrg(file_path, outmod):
    """
    Reorganize file path information into a DataFrame.

    Parameters:
    file_path (str): The file path to process.
    outmod (str): The modality to be output.

    Returns:
    pandas.DataFrame: A DataFrame containing the processed file path information.
    """
    temp = file_path.split(os.sep)[1:]  # Split the path and remove the first empty element
    mydf = pd.DataFrame({
        'id': [temp[0]],
        'dt': [temp[1]],
        'modality': [outmod],
        'imageID': ['000'],
        'filename': [file_path]
    })
    return mydf

# Using glob to find files similar to R's Sys.glob
suids = glob.glob("images/PEDS*/*/*/*t1.nii.gz")
# Processing the first file (if it exists) and printing the DataFrame
if suids:
    for x in range(len(suids)):
        t1df = peds2nrg(suids[x], 'T1w')
        print(t1df)
        mynrg = antspymm.nrg_format_path('PTBP', 
            t1df['id'].iloc[0], t1df['dt'].iloc[0], 
            t1df['modality'].iloc[0], '000', separator='-')
        create_directory_and_copy_file(t1df['filename'].iloc[0], mynrg+'.nii.gz' )

suids = glob.glob("images/PEDS*/*/*/*bold*.nii.gz")
# Processing the first file (if it exists) and printing the DataFrame
if suids:
    for x in range(len(suids)):
        t1df = peds2nrg(suids[x], 'rsfMRI')
        print(t1df)
        mynrg = antspymm.nrg_format_path('PTBP', 
            t1df['id'].iloc[0], t1df['dt'].iloc[0], 
            t1df['modality'].iloc[0], '000', separator='-')
        create_directory_and_copy_file(t1df['filename'].iloc[0], mynrg+'.nii.gz' )


suids = glob.glob("images/PEDS*/*/*/*pcasl*.nii.gz")
# Processing the first file (if it exists) and printing the DataFrame
if suids:
    for x in range(len(suids)):
        t1df = peds2nrg(suids[x], 'perf')
        print(t1df)
        mynrg = antspymm.nrg_format_path('PTBP', 
            t1df['id'].iloc[0], t1df['dt'].iloc[0], 
            t1df['modality'].iloc[0], '000', separator='-')
        create_directory_and_copy_file(t1df['filename'].iloc[0], mynrg+'.nii.gz' )

# raise ValueError("FIXME DTI")
for num in ['0011','0014','0017','0020']:
    lsearch=num+"_DTI"
    suids = glob.glob("images/PEDS*/*/*/*"+lsearch+"*.nii.gz")
    # Processing the first file (if it exists) and printing the DataFrame
    if suids:
        for x in range(len(suids)):
            t1df = peds2nrg(suids[x], 'DTI')
            print(t1df)
            mynrg = antspymm.nrg_format_path('PTBP', 
                t1df['id'].iloc[0], t1df['dt'].iloc[0], 
                t1df['modality'].iloc[0], num, separator='-')
            create_directory_and_copy_similar_files(t1df['filename'].iloc[0], mynrg+'.nii.gz' )
