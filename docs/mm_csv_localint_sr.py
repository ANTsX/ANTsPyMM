#!/usr/bin/env python3
################################################################
#  for easier to access data with a full mm_csv example, see:  #
#  github.com:stnava/ANTPD_antspymm                            #
################################################################
import os
seed = 42  #
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# this is important for reading models via siq.read_srmodel
nthreads = str(48) # for much faster performance and good reproducibility
os.environ["TF_NUM_INTEROP_THREADS"] = nthreads
os.environ["TF_NUM_INTRAOP_THREADS"] = nthreads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
import random
import numpy as np
random.seed(seed)
np.random.seed(seed)
from os.path import exists
import signal
import urllib.request
import zipfile
import tempfile
from pathlib import Path
from tqdm import tqdm
import siq
import antspynet
#####################
REQUIRED_FILES = [
    "PPMI/101018/20210412/T1w/1496225/PPMI-101018-20210412-T1w-1496225.nii.gz",
    "PPMI/101018/20210412/DTI_LR/1496234/PPMI-101018-20210412-DTI_LR-1496234.nii.gz"
]
#####################
# make sure we can read the disc
print("read the SR model ")
mfn=os.path.expanduser('~/.antspymm/siq_default_sisr_2x2x2_2chan_featgraderL6_best.keras')
mfn=os.path.expanduser('~/.antspymm/siq_smallshort_train_2x2x2_1chan_featgraderL6_best.keras')
mdl, mdlshape = siq.read_srmodel(mfn)
print("read the SR model done")
#####################
def _validate_required_files(base_dir, required_files):
    for rel_path in required_files:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.isfile(full_path):
            print(f"❌ Missing required file: {rel_path}")
            return False
    return True

def _download_with_progress(url, destination):
    with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
        total = int(response.getheader('Content-Length', 0))
        with tqdm(total=total, unit='B', unit_scale=True, desc="Downloading", ncols=80) as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))

def find_data_dir(candidate_paths=None, max_tries=5, timeout=22, allow_download=None, required_files=REQUIRED_FILES):
    """
    Attempts to locate or download the ANTsPyMM testing dataset.

    Parameters
    ----------
    candidate_paths : list of str or None
        Directories to search for the data. If None, uses sensible defaults.
    max_tries : int
        Number of chances to enter a valid path manually.
    timeout : int
        Seconds to wait for user input before timing out.
    allow_download : None | str
        If not None, will download to {allow_download}/nrgdata_test if needed.
    required_files : list of str
        Relative paths that must exist inside the data directory.

    Returns
    -------
    str
        Path to a valid data directory.
    """
    if candidate_paths is None:
        candidate_paths = [
            "~/Downloads/temp/shortrun/nrgdata_test",
            "~/Downloads/ANTsPyMM_testing_data/nrgdata_test",
            "~/data/ppmi/nrgdata_test",
            "/mnt/data/nrgdata_test"
        ]

    # First, search known paths
    for path in candidate_paths:
        full_path = os.path.expanduser(path)
        if os.path.isdir(full_path) and _validate_required_files(full_path, required_files):
            print(f"✅ Found valid data directory: {full_path}")
            return full_path

    # Handle automatic download
    if isinstance(allow_download, str):
        base_dir = os.path.expanduser(allow_download)
        target_dir = os.path.join(base_dir, "nrgdata_test")
        if not os.path.isdir(target_dir) or not _validate_required_files(target_dir, required_files):
            print(f"📥 Will download data to: {target_dir}")
            url = "https://figshare.com/ndownloader/articles/29391236/versions/1"
            os.makedirs(base_dir, exist_ok=True)
            zip_path = os.path.join(tempfile.gettempdir(), "antspymm_testdata.zip")

            try:
                _download_with_progress(url, zip_path)
                print("📦 Extracting...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(base_dir)
                print(f"✅ Extracted to {target_dir}")
            except Exception as e:
                raise RuntimeError(f"❌ Download or extraction failed: {e}")

        if not _validate_required_files(target_dir, required_files):
            raise RuntimeError(f"❌ Downloaded data is missing required files in {target_dir}")
        return target_dir

    # Timeout handler for POSIX
    def timeout_handler(signum, frame):
        raise TimeoutError("⏳ No input received in time.")

    if os.name == 'posix':
        signal.signal(signal.SIGALRM, timeout_handler)

    # Manual user prompt
    print("🔍 Could not find valid data. You may enter a directory manually.")
    print("🔗 Dataset info: https://figshare.com/articles/dataset/ANTsPyMM_testing_data/29391236")

    for attempt in range(1, max_tries + 1):
        try:
            if os.name == 'posix':
                signal.alarm(timeout)
            user_input = input(f"⏱️ Attempt {attempt}/{max_tries} — Enter data directory (or 'q' to quit): ").strip()
            if os.name == 'posix':
                signal.alarm(0)

            if user_input.lower() == 'q':
                break

            path = os.path.expanduser(user_input)
            if os.path.isdir(path) and _validate_required_files(path, required_files):
                print(f"✅ Using user-provided directory: {path}")
                return path
            else:
                print("❌ Invalid or incomplete directory.")

        except TimeoutError as e:
            raise RuntimeError(str(e))
        except KeyboardInterrupt:
            raise RuntimeError("User interrupted execution. Exiting.")

    raise RuntimeError("❗ No valid data directory found and download not permitted.")

candidate_rdirs = [
    "~/Downloads/nrgdata_test/",
    "~/Downloads/temp/nrgdata_test/",
    "~/nrgdata_test/",
    "~/data/ppmi/nrgdata_test/",
    "/mnt/data/ppmi_testing/nrgdata_test/"]
########################################################################
rdir = find_data_dir( candidate_rdirs, allow_download="~/Downloads" )
print(f"Using data directory: {rdir}")
########################################################################
import numpy as np
import glob as glob
import antspymm
import ants
import antspyt1w
import random
import re
print("Begin template loading")
tlrfn = antspyt1w.get_data('T_template0_LR', target_extension='.nii.gz' )
tfn = antspyt1w.get_data('T_template0', target_extension='.nii.gz' )
templatea = ants.image_read( tfn )
templatea = ( templatea * antspynet.brain_extraction( templatea, 't1' ) ).iMath( "Normalize" )
templatealr = ants.image_read( tlrfn )
print("done template loading")

if __name__ == '__main__':
    repro=True
    repro=False
    mydir = rdir + "PPMI/"
    if not exists(repro):
        repro = False
    if not repro:
        outdir = re.sub( 'nrgdata_test', 'antspymmoutput_sr1c_th'+nthreads, rdir )
    else:
        outdir = re.sub( 'nrgdata_test', 'antspymmoutput_sr1c_th'+nthreads+'_repro', rdir )
    ################################
    print( " outdir = " + outdir ) #
    ################################
    import antspymm #####
    import pandas as pd #
    import glob as glob #
    t1fn=glob.glob(mydir+"101018/20210412/T1w/1496225/*.nii.gz")
    if len(t1fn) > 0:
        t1fn=t1fn[0]
        testimg = ants.image_read( t1fn )
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
        template = ants.image_read("~/.antspymm/PPMI_template0.nii.gz").reorient_image2("LPI")
        bxt = ants.image_read("~/.antspymm/PPMI_template0_brainmask.nii.gz").reorient_image2("LPI")
        template = template * bxt
        template = ants.crop_image( template, ants.iMath( bxt, "MD", 12 ) )
        mfnmm = re.sub( "2x2x2", "bestup", mfn )
        mmrun = antspymm.mm_csv( studycsv2,
            srmodel_T1 = mfn,
            srmodel_NM = mfnmm,
            srmodel_DTI = mfnmm,
            normalization_template=template,
            normalization_template_output='ppmi',
            normalization_template_transform_type='antsRegistrationSyNQuickRepro[s]',
            normalization_template_spacing=[1,1,1]  )
        pdir = str(studycsv2['outputdir'][0])
        merged=antspymm.aggregate_antspymm_results_sdf( studycsv2,
            subject_col='subjectID', date_col='date', image_col='imageID',  base_path=pdir,
            splitsep='-', idsep='-', wild_card_modality_id=True, second_split=False, verbose=True )
        print(merged.shape)
        index=0
        outfn = pdir + studycsv2['projectID'][index]+'-'+ studycsv2['subjectID'][index]+'-'+ studycsv2['date'][index]+'-'+studycsv2['imageID'][index]+'-th'+str(nthreads)+'-mmwide.csv'
        merged.to_csv( outfn )
    else:
        print("We searched "+mydir+"101018/20210412/T1w/1496225/*.nii.gz")
        print("T1w data is missing: see github.com:stnava/ANTPD_antspymm for a full integration study and container with more easily accessible data")
