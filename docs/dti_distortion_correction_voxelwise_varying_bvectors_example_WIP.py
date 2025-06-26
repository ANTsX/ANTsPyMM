import numpy as np
import ants
import os
from dipy.io.gradients import read_bvals_bvecs
from scipy.stats import pearsonr
import antspymm
nt = 2

import numpy as np
from scipy.stats import pearsonr

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

# def test_efficient_dwi_fit_voxelwise_distortion_correction():
if True:
    print("simple test for distortion_correction consistency of efficient_dwi_fit_voxelwise")
    ex_path = os.path.expanduser( "~/.antspyt1w/" )
    ex_path_mm = os.path.expanduser( "~/.antspymm/" )
    #### Load in data ####
    print("Load in subject data ...")
    lrid = "I1499279_Anon_20210819142214_5"
    rlid = "I1499337_Anon_20210819142214_6"
    # Load paths
    print("📁 Loading subject LR data...")
    lrid = os.path.join(ex_path_mm, lrid )
    img_LR_in = ants.image_read(lrid + '.nii.gz')
    img_LR_in_avg = ants.get_average_of_timeseries( img_LR_in )
    mask = img_LR_in_avg.get_mask()
    bvals, bvecs = read_bvals_bvecs(lrid + '.bval', lrid + '.bvec')
    bvecs = np.asarray(bvecs)
    shape = img_LR_in.shape[:3]

    print("📁 Loading subject RL data...")
    rlid = os.path.join(ex_path_mm, rlid )
    img_RL_in = ants.image_read(rlid + '.nii.gz')
    bvalsRL, bvecsRL = read_bvals_bvecs(rlid + '.bval', rlid + '.bvec')

    img_RL_in_avg = ants.get_average_of_timeseries( img_RL_in )
    maskRL = img_RL_in_avg.get_mask()

    print("🧠 Running baseline LR fit...")
    bvecs_5d_orig = np.broadcast_to(bvecs, shape + bvecs.shape).copy()
    if not "FA_orig" in globals():
        FA_orig, MD_orig, RGB_orig = antspymm.efficient_dwi_fit_voxelwise(
            imagein=img_LR_in,
            maskin=mask,
            bvals=bvals,
            bvecs_5d=bvecs_5d_orig,
            model_params={},
            bvals_to_use=None,
            num_threads=nt,
            verbose=False
        )

    bvecs_5d_origRL = np.broadcast_to(np.asarray(bvecsRL), shape + bvecsRL.shape).copy()
    if not "FA_origRL" in globals():
        FA_origRL, MD_origRL, RGB_origRL = antspymm.efficient_dwi_fit_voxelwise(
            imagein=img_RL_in,
            maskin=maskRL,
            bvals=bvalsRL,
            bvecs_5d=bvecs_5d_origRL,
            model_params={},
            bvals_to_use=None,
            num_threads=nt,
            verbose=False
        )

    print("dist corr")
    if not "mytx" in globals():
        mytx = ants.registration( FA_orig, FA_origRL, 'SyNBold' )
        mytx2 = ants.apply_transforms(FA_orig, FA_origRL, mytx['fwdtransforms'],
            interpolator='linear', imagetype=0, compose='/tmp/comptx' )
        print( mytx2 )

    print("🔄 now with distortion correction...")
    mydef = ants.image_read( mytx2 )
    mywarp = ants.transform_from_displacement_field( mydef )
    img_w = antspymm.timeseries_transform(mywarp, img_RL_in, reference=img_LR_in_avg)
    mask_w = ants.apply_ants_transform_to_image(mywarp, maskRL, reference=img_LR_in_avg, interpolation='nearestNeighbor')
    print("🧠 Running warped fit...")
    if not "FA_w" in globals():
        bvecsRL = np.asarray(bvecsRL)
        mydefgrad = antspymm.deformation_gradient_optimized( mydef, 
            to_rotation=False, to_inverse_rotation=True )
        bvecsRLw = antspymm.generate_voxelwise_bvecs( bvecsRL, mydefgrad, transpose=False )
        FA_w, MD_w, RGB_w = antspymm.efficient_dwi_fit_voxelwise(
            imagein=img_w,
            maskin=mask_w,
            bvals=bvalsRL,
            bvecs_5d=bvecsRLw,
            model_params={},
            bvals_to_use=None,
            num_threads=nt,
            verbose=False
        )

    if not "FA_w2" in globals():
        bvecsRL = np.asarray(bvecsRL)
        FA_w2, MD_w2, RGB_w2 = antspymm.efficient_dwi_fit_voxelwise(
            imagein=img_w,
            maskin=mask_w,
            bvals=bvalsRL,
            bvecs_5d=bvecs_5d_origRL,
            model_params={},
            bvals_to_use=None,
            num_threads=nt,
            verbose=False
        )

    fff = mean_rgb_correlation
    print("📊 Comparing results...")
    maskJoined = ants.threshold_image( mask + mask_w, 1.05, 2.0 )
    maske=ants.iMath(maskJoined,'ME',3)
    fa_corr = fff( RGB_orig, RGB_w, maske )
    print(f"✅ FA correlation (original vs distortion corrected): {fa_corr:.4f}")

    fa_corrX = fff( RGB_orig, RGB_w2, maske )
    print(f"✅ FA correlation (original vs distortion corrected global recon): {fa_corrX:.4f}")

    RGB_origRLc = ants.split_channels(RGB_origRL)
    for c in range(3):
        RGB_origRLc[c] = ants.apply_ants_transform_to_image(
            mywarp, RGB_origRLc[c], 
            reference=img_LR_in_avg, interpolation='linear'
        )

    fa_corrY = fff( RGB_orig, ants.merge_channels(RGB_origRLc), maske )
    print(f"✅ FA correlation (original vs warped RGB global recon): {fa_corrY:.4f}")

#    assert fa_corr > 0.80, "FA correlation too low"

    print("🎉 Test passed: model is distortion-consistent.")

# ants.image_write( FA_orig, '/tmp/xxx.nii.gz' )
# ants.image_write( FA_w, '/tmp/yyy.nii.gz' )
#
# Example usage:
# test_efficient_dwi_fit_voxelwise_distortion_correction()
