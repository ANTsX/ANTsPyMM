import numpy as np
import ants
import os
from dipy.io.gradients import read_bvals_bvecs
from scipy.stats import pearsonr
import antspymm
nt = 2
def extract_masked_values(image, mask):
    return image.numpy()[mask.numpy() > 0]

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
    img_RL_in_avg = ants.get_average_of_timeseries( img_RL_in )
    maskRL = img_RL_in_avg.get_mask()

    print("dist corr")
    if not "mytx" in globals():
        mytx = ants.registration( img_LR_in_avg, img_RL_in_avg, 'SyNBold' )
        mytx2 = ants.apply_transforms(img_LR_in_avg, img_RL_in_avg, mytx['fwdtransforms'], interpolator='linear', imagetype=0, compose='/tmp/comptx' )
        print( mytx2 )

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

    print("🔄 now with distortion correction...")
    mydef = ants.image_read( mytx2 )
    mywarp = ants.transform_from_displacement_field( mydef )
    img_w = antspymm.timeseries_transform(mywarp, img_RL_in, reference=img_LR_in_avg)
    mask_w = ants.apply_ants_transform_to_image(mywarp, maskRL, reference=img_LR_in_avg, interpolation='nearestNeighbor')
    print("🧠 Running warped fit...")
    if not "FA_w" in globals():
        bvalsRL, bvecsRL = read_bvals_bvecs(rlid + '.bval', rlid + '.bvec')
        bvecsRL = np.asarray(bvecsRL)
        mydefgrad = antspymm.deformation_gradient_optimized( mydef, 
            to_rotation=False, to_inverse_rotation=True )
        bvecsRLw = antspymm.generate_voxelwise_bvecs( bvecsRL, mydefgrad )
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

    # FA_w_back = ants.apply_transforms( img_LR_in_avg, FA_w, mytx['invtransforms'], whichtoinvert=[True,False] )
    print("📊 Comparing results...")
    maske=ants.iMath(mask,'ME',2)
    fa1 = extract_masked_values(FA_orig, maske)
    fa2 = extract_masked_values(FA_w, maske)
    fa_corr, _ = pearsonr(fa1, fa2)

    print(f"✅ FA correlation (original vs rotated): {fa_corr:.4f}")
    
    assert fa_corr > 0.80, "FA correlation too low"

    print("🎉 Test passed: model is distortion-consistent.")

# Example usage:
# test_efficient_dwi_fit_voxelwise_distortion_correction()
