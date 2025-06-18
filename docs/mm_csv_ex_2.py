##################################################################
# This script processes neuroimaging data in NRG format using antspymm.
# NRG format: https://github.com/stnava/biomedicalDataOrganization
##################################################################

import os
import glob
import numpy as np
import pandas as pd
import ants
import antspymm

# ---------------------------
# Configuration
# ---------------------------
nthreads = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = nthreads
os.environ["TF_NUM_INTRAOP_THREADS"] = nthreads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads

# Base directory in NRG format
out_dir = os.path.expanduser("./example/processedX/")
base_dir = os.path.expanduser("./nrgdata/data/")
projid="PPMI"
participant_num = "182341"
participant_date = "20230111"
t1id="1681010"
participant_path = os.path.join(base_dir, projid, participant_num, participant_date)

# ---------------------------
# Locate image files
# ---------------------------
def get_first_file(pattern, required=True):
    files = glob.glob(pattern)
    if required and not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files[0] if files else None

t1_file = get_first_file(os.path.join(participant_path, "T1w/"+t1id+"/*.nii.gz"))
flair_file = get_first_file(os.path.join(participant_path, "T2Flair/*/*.nii.gz"))
dti_files = glob.glob(os.path.join(participant_path, "DTI*/*/*.nii.gz"))
rsf_files = glob.glob(os.path.join(participant_path, "rsfMRI*/*/*1681014*.nii.gz"))+glob.glob(os.path.join(participant_path, "rsfMRI*/*/*1681015*.nii.gz"))
nm_files = glob.glob(os.path.join(participant_path, "NM2DMT/*/*.nii.gz"))

# ---------------------------
# Generate multi-modal dataframe
# ---------------------------
mm_df = antspymm.generate_mm_dataframe(
    projectID='PPMI',
    subjectID=participant_num,
    date=participant_date,
    imageUniqueID=t1id,
    modality="T1w",
    source_image_directory=base_dir,
    output_image_directory=out_dir,
    t1_filename=t1_file,
    flair_filename=flair_file,
    rsf_filenames=rsf_files,
    dti_filenames=dti_files,
    nm_filenames=nm_files
)

# Drop NA columns and run analysis
mm_df_clean = mm_df.dropna(axis=1)
print(mm_df_clean)

# Run antspymm pipeline
mm_result = antspymm.mm_csv(mm_df_clean)