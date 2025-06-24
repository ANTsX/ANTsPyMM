##################################################################
# for easier to access data with a full mm_csv example, see:
# github.com:stnava/ANTPD_antspymm
##################################################################
import os
from os.path import exists
import signal

def find_data_dir(candidate_paths, max_tries=5, timeout=8):
    """
    Search for the first existing directory in a list. If none are found,
    prompt the user to enter a path (with timeout) up to `max_tries` times.

    Parameters
    ----------
    candidate_paths : list of str
        Paths to try in order.
    max_tries : int
        Maximum number of manual attempts.
    timeout : int
        Time (in seconds) to wait for user input before exiting.

    Returns
    -------
    str
        A valid existing directory path.

    Raises
    ------
    RuntimeError
        If no path is found or selected within the allowed attempts.
    """
    # Try provided paths first
    for path in candidate_paths:
        expanded = os.path.expanduser(path)
        if os.path.isdir(expanded):
            print(f"✅ Found data directory: {expanded}")
            return expanded

    # Setup timeout handler (POSIX only)
    def timeout_handler(signum, frame):
        raise TimeoutError("⏳ No input received in time. Exiting.")

    if os.name == 'posix':
        signal.signal(signal.SIGALRM, timeout_handler)

    print("📦 None of the default data directories were found.")
    print("🔗 Official dataset (if needed): https://figshare.com/articles/dataset/ANTsPyMM_testing_data/29391236")

    tries = 0
    while tries < max_tries:
        tries += 1
        try:
            if os.name == 'posix':
                signal.alarm(timeout)
            user_input = input(f"⏱️ Attempt {tries}/{max_tries} — Enter valid directory (or 'q' to quit): ").strip()
            if os.name == 'posix':
                signal.alarm(0)

            if user_input.lower() == 'q':
                raise RuntimeError("User aborted directory selection.")

            expanded_input = os.path.expanduser(user_input)
            if os.path.isdir(expanded_input):
                print(f"✅ Using user-provided directory: {expanded_input}")
                return expanded_input
            else:
                print(f"❌ '{expanded_input}' is not a valid directory.")

        except TimeoutError as e:
            raise RuntimeError(str(e))
        except KeyboardInterrupt:
            raise RuntimeError("User interrupted execution. Exiting.")

    raise RuntimeError("❗ No valid directory found after maximum attempts.")

rdir = find_data_dir([
    "~/Downloads/nrgdata_test/",
    "~/Downloads/temp/shortrun/nrgdata_test/",
    "~/nrgdata_test/",
    "~/data/ppmi/nrgdata_test/",
    "/mnt/data/ppmi_testing/nrgdata_test/"
])
print(f"Using data directory: {rdir}")

nthreads = str(8)
os.environ["TF_NUM_INTEROP_THREADS"] = nthreads
os.environ["TF_NUM_INTRAOP_THREADS"] = nthreads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
import numpy as np
import glob as glob
import antspymm
import ants
import random
import re
mydir = rdir + "PPMI/"
outdir = re.sub( 'nrgdata_test', 'antspymmoutput', rdir )
################
import antspymm
import pandas as pd
import glob as glob
t1fn=glob.glob(mydir+"101018/20210412/T1w/1496225/*.nii.gz")
if len(t1fn) > 0:
    t1fn=t1fn[0]
    print("Begin " + t1fn)
    flfn=glob.glob(mydir+"101018/20210412/T2Flair/*/*.nii.gz")[0]
    dtfn=glob.glob(mydir+"101018/20210412/DTI*/*/*.nii.gz")
    rsfn=glob.glob(mydir+"101018/20210412/rsfMRI*/*/*.nii.gz")
    nmfn=glob.glob(mydir+"101018/20210412/NM2DMT/*/*.nii.gz")
    studycsv = antspymm.generate_mm_dataframe( 
        projectID='PPMI',
        subjectID='101018', 
        date='20210412', 
        imageUniqueID='1496225', 
        modality='T1w', 
        source_image_directory=rdir, 
        output_image_directory=outdir, 
        t1_filename = t1fn,
        flair_filename=flfn,
        rsf_filenames=rsfn,
        dti_filenames=dtfn,
        nm_filenames=nmfn
    )
    studycsv2 = studycsv.dropna(axis=1)
    print( studycsv2 )
    mmrun = antspymm.mm_csv( studycsv2  )
else:
    print("T1w data is missing: see github.com:stnava/ANTPD_antspymm for a full integration study and container with more easily accessible data")