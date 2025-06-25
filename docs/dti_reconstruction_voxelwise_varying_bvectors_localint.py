import numpy as np
import ants
import os
from dipy.io.gradients import read_bvals_bvecs
from scipy.stats import pearsonr
import antspymm
nt = 2
def read_bvecs_rotated(bvec_file, rotmat):
    bvecs = np.loadtxt(bvec_file)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T
    rotated_bvecs = (rotmat @ bvecs).T
    return rotated_bvecs

def broadcast_bvecs_voxelwise(rotated_bvecs, shape):
    return np.broadcast_to(rotated_bvecs, shape + rotated_bvecs.shape).copy()

def extract_masked_values(image, mask):
    return image.numpy()[mask.numpy() > 0]

def test_efficient_dwi_fit_voxelwise_rotation_consistency():
    print("simple test for rotation consistency of efficient_dwi_fit_voxelwise")
    ex_path = os.path.expanduser( "~/.antspyt1w/" )
    ex_path_mm = os.path.expanduser( "~/.antspymm/" )
    JHU_atlas = ants.image_read( ex_path + 'JHU-ICBM-FA-1mm.nii.gz' ) # Read in JHU atlas
    JHU_labels = ants.image_read( ex_path + 'JHU-ICBM-labels-1mm.nii.gz' ) # Read in JHU labels
    #### Load in data ####
    print("Load in subject data ...")
    lrid = ex_path_mm + "I1499279_Anon_20210819142214_5"
    rlid = ex_path_mm + "I1499337_Anon_20210819142214_6"
    t1id = ex_path_mm + "t1_rand.nii.gz"
    # Load paths
    print("📁 Loading subject data...")
    lrid = os.path.join(ex_path_mm, "I1499279_Anon_20210819142214_5")
    img_LR_in = ants.image_read(lrid + '.nii.gz')
    img_LR_in_avg = ants.get_average_of_timeseries( img_LR_in )
    mask = img_LR_in_avg.get_mask()

    bvals, bvecs = read_bvals_bvecs(lrid + '.bval', lrid + '.bvec')
    bvecs = np.asarray(bvecs)
    shape = img_LR_in.shape[:3]

    print("🧠 Running baseline (unrotated) fit...")
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

    print("🔄 Applying known rotation...")
    # maxtrans = 10.0
    # rotator = ants.contrib.RandomRotate3D((-maxtrans, maxtrans), reference=img_LR_in_avg)
    rotator=ants.contrib.Rotate3D(rotation=(0,0,90), reference=img_LR_in_avg )
    rotation = rotator.transform()
    img_rotated_ref = ants.apply_ants_transform_to_image(rotation, img_LR_in_avg, reference=img_LR_in_avg)
    img_rotated = antspymm.timeseries_transform(rotation, img_LR_in, reference=img_LR_in_avg)
    mask_rotated = ants.apply_ants_transform_to_image(rotation, mask, reference=img_LR_in_avg, interpolation='nearestNeighbor')
    rotmat = ants.get_ants_transform_parameters(rotation).reshape((4, 3))[:3, :3]
    bvecs_rotated = read_bvecs_rotated(lrid + '.bvec', rotmat)
    bvecs_5d_rot = broadcast_bvecs_voxelwise(bvecs_rotated, shape)

    print("🧠 Running rotated fit...")
    if not "FA_rot" in globals():
        FA_rot, MD_rot, RGB_rot = antspymm.efficient_dwi_fit_voxelwise(
            imagein=img_rotated,
            maskin=mask_rotated,
            bvals=bvals,
            bvecs_5d=bvecs_5d_rot,
            model_params={},
            bvals_to_use=None,
            num_threads=nt,
            verbose=False
        )

    FA_rot_back = ants.apply_ants_transform_to_image(
        rotation.invert(),
        FA_rot,
        reference = img_LR_in_avg,
        )
    MD_rot_back = ants.apply_ants_transform_to_image(
        rotation.invert(),
        MD_rot,
        reference = img_LR_in_avg,
        )
    print("📊 Comparing results...")
    maske=ants.iMath(mask,'ME',2)
    fa1 = extract_masked_values(FA_orig, maske)
    fa2 = extract_masked_values(FA_rot_back, maske)
    md1 = extract_masked_values(MD_orig, maske)
    md2 = extract_masked_values(MD_rot_back, maske)

    fa_corr, _ = pearsonr(fa1, fa2)
    md_corr, _ = pearsonr(md1, md2)

    print(f"✅ FA correlation (original vs rotated): {fa_corr:.4f}")
    print(f"✅ MD correlation (original vs rotated): {md_corr:.4f}")
    
    assert fa_corr > 0.80, "FA correlation too low"
    assert md_corr > 0.80, "MD correlation too low"

    print("🎉 Test passed: model is rotation-consistent with voxelwise bvecs.")

# Example usage:
test_efficient_dwi_fit_voxelwise_rotation_consistency()
