import numpy as np
import ants
import os
from dipy.io.gradients import read_bvals_bvecs
from scipy.stats import pearsonr
import antspymm
import numpy as np
from scipy.stats import pearsonr
nt=8

##################################################################
# for easier to access data with a full mm_csv example, see:
# github.com:stnava/ANTPD_antspymm
##################################################################
from os.path import exists
import os
import signal
import urllib.request
import zipfile
import tempfile
from pathlib import Path
from tqdm import tqdm
import antspynet

REQUIRED_FILES = [
    "PPMI/101018/20210412/T1w/1496225/PPMI-101018-20210412-T1w-1496225.nii.gz",
    "PPMI/101018/20210412/DTI_LR/1496234/PPMI-101018-20210412-DTI_LR-1496234.nii.gz"
]

def broadcast_bvecs_voxelwise(rotated_bvecs, shape):
    return np.broadcast_to(rotated_bvecs, shape + rotated_bvecs.shape).copy()


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


rdir = find_data_dir( candidate_rdirs, allow_download="~/Downloads" )
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

def read_bvecs_rotated(bvec_file, rotmat):
    bvecs = np.loadtxt(bvec_file)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T
    rotated_bvecs = (rotmat @ bvecs).T
    return rotated_bvecs

def mean_rgb_correlation(img1, img2, mask):
    """
    Compute the mean correlation between two RGB images.

    Parameters
    ----------
    img1 : np.ndarray
        First RGB image as a (H, W, 3) NumPy array.
    img2 : np.ndarray
        Second RGB image as a (H, W, 3) NumPy array.

    Returns
    -------
    float
        Mean Pearson correlation across the three RGB channels.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")
    correlations = []
    img1c = ants.split_channels(img1)
    img2c = ants.split_channels(img2)   
    for c in range(3):  # R, G, B
        x = extract_masked_values( img1c[c], mask)
        y = extract_masked_values( img2c[c], mask)
        if np.std(x) == 0 or np.std(y) == 0:
            corr = 0.0  # Handle flat images
        else:
            corr, _ = pearsonr(x, y)
        correlations.append(corr)
    return np.mean(correlations)

import numpy as np
import ants

def mean_rgb_mae(img1, img2, mask):
    """
    Compute the mean absolute error (MAE) between two RGB images.

    Parameters
    ----------
    img1 : np.ndarray
        First RGB image as a (H, W, 3) NumPy array.
    img2 : np.ndarray
        Second RGB image as a (H, W, 3) NumPy array.
    mask : ants.ANTsImage
        Binary mask defining valid pixels for error calculation.

    Returns
    -------
    float
        Mean absolute error across the three RGB channels.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")
    
    mae_values = []
    img1c = ants.split_channels(img1)
    img2c = ants.split_channels(img2)   
    
    for c in range(3):  # R, G, B
        x = extract_masked_values(img1c[c], mask)
        y = extract_masked_values(img2c[c], mask)
        mae = np.mean(np.abs(x - y))
        mae_values.append(mae)
    
    return np.mean(mae_values)

def extract_masked_values(image, mask):
    return image.numpy()[mask.numpy() > 0]

import numpy as np

def verify_unit_bvecs(bvecs, tol=1e-5):
    """
    Verifies that each b-vector has unit norm within a tolerance.
    
    Parameters
    ----------
    bvecs : array-like, shape (N, 3)
        Array of b-vectors, one per diffusion direction.
    tol : float
        Tolerance for unit norm.

    Returns
    -------
    is_unit : np.ndarray, shape (N,)
        Boolean array indicating if each b-vector is unit norm.
    norms : np.ndarray, shape (N,)
        Norms of each b-vector.
    """
    bvecs = np.asarray(bvecs)
    norms = np.linalg.norm(bvecs, axis=1)
    is_unit = np.abs(norms - 1) < tol
    return is_unit, norms

mydir = rdir + "PPMI/"
outdir = re.sub( 'nrgdata_test', 'antspymmoutput', rdir )
import glob as glob

t1fn=glob.glob(mydir+"101018/20210412/T1w/1496225/*.nii.gz")
if len(t1fn) > 0:
    t1fn=t1fn[0]
    print("Begin " + t1fn)
    dtfn=glob.glob(mydir+"101018/20210412/DTI*/*/*.nii.gz")
    dtfn.sort()

import re

# def test_efficient_dwi_fit_voxelwise_distortion_correction():
if len(dtfn) > 0:
    img_LR_in = ants.image_read(dtfn[0])
    img_LR_in_avg = ants.get_average_of_timeseries( img_LR_in )
    mask = img_LR_in_avg.get_mask()
    bvalfn = re.sub( 'nii.gz', 'bval', dtfn[0] )
    bvecfn = re.sub( 'nii.gz', 'bvec', dtfn[0] )
    if not exists(bvalfn) or not exists(bvecfn):
        raise RuntimeError(f"Required bval/bvec files not found: {bvalfn}, {bvecfn}")
    print(f"📁 Loading subject LR data from {bvalfn} ")
    bvals, bvecs = read_bvals_bvecs(bvalfn, bvecfn)
    bvecs = np.asarray(bvecs)
    shape = img_LR_in.shape[:3]

    print("📁 Loading subject T1 data...")
    t1w = ants.image_read(t1fn)
    t1w = ants.resample_image(t1w, [2, 2, 2], use_voxels=False)
    bxt = antspynet.brain_extraction(t1w, modality='t1', verbose=False).threshold_image(0.5, 1.5)

    if not "mytx" in globals():
        dwianat = ants.slice_image( img_LR_in, idx=0, axis=3)
        mytx = ants.registration( t1w, dwianat, 'SyNCC', syn_metric='CC', syn_sampling=2, total_sigma=0.5 )
        mytx2 = ants.apply_transforms(t1w, img_LR_in_avg, mytx['fwdtransforms'],
            interpolator='linear', imagetype=0, compose='/tmp/comptxDT2T1' )
        print( mytx2 )

    print("🔄 now with distortion correction...")
    mydef = ants.image_read( mytx2 )
    mywarp = ants.transform_from_displacement_field( mydef )
    img_w = antspymm.timeseries_transform(mywarp, img_LR_in, reference=t1w)
    print("🧠 Running warped fit...")
    
    if not "FA_w" in globals():
        mydefgrad = antspymm.deformation_gradient_optimized( mydef, 
            to_rotation=False, to_inverse_rotation=True )
        bvecsdc = antspymm.distortion_correct_bvecs( bvecs, mydefgrad, t1w.direction, img_LR_in_avg.direction )
        FA_w, MD_w, RGB_w = antspymm.efficient_dwi_fit_voxelwise(
            imagein=img_w,
            maskin=bxt,
            bvals=bvals,
            bvecs_5d=bvecsdc,
            model_params={},
            bvals_to_use=None,
            num_threads=nt,
            verbose=False
        )

    if not "FA_w2" in globals():
        myrig = ants.registration( t1w, dwianat, 'Rigid')
        rigtx = ants.read_transform( myrig['fwdtransforms'][0] )
        img_w2 = antspymm.timeseries_transform( rigtx, img_LR_in, reference=t1w)
        FA_w2, MD_w2, RGB_w2 = antspymm.efficient_dwi_fit_voxelwise(
            imagein=img_w2,
            maskin=bxt,
            bvals=bvals,
            bvecs_5d=broadcast_bvecs_voxelwise(bvecs, t1w.shape),
            model_params={},
            bvals_to_use=None,
            num_threads=nt,
            verbose=False
        )

    print("📊 Comparing results...")
    maske=ants.iMath(bxt,'ME',3)
    fa_corr = mean_rgb_correlation( RGB_w, RGB_w2, maske )
    print(f"✅ Direction-weighted FA correlation (original vs distortion corrected): {fa_corr:.4f}")

ants.image_write( t1w, '/tmp/t1w.nii.gz' )
ants.image_write( RGB_w, '/tmp/rgbw.nii.gz' )
ants.image_write( RGB_w2, '/tmp/rgbw2.nii.gz' )
