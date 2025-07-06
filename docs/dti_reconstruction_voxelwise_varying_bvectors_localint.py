import numpy as np
import ants
import os
from dipy.io.gradients import read_bvals_bvecs
from scipy.stats import pearsonr
import antspymm
import matplotlib.pyplot as plt
nt = 8
# amount or rotation around x, y, z-axis
degrotx = 15
degroty = 15
degrotz = 15


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


def plot_correlation(x, y, xlabel="Image 1", ylabel="Image 2", title_prefix="Correlation", point_alpha=0.3):
    """
    Plot the correlation between two 1D arrays and display Pearson r.

    Parameters
    ----------
    x : array-like
        First set of values.
    y : array-like
        Second set of values.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title_prefix : str
        Title prefix, will append r value.
    point_alpha : float
        Transparency of scatter plot points.

    Returns
    -------
    float
        Pearson correlation coefficient (r).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")

    r, _ = pearsonr(x, y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, alpha=point_alpha, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} (r = {r:.3f})")
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', linewidth=1)  # Identity line
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    return r

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

# def test_efficient_dwi_fit_voxelwise_rotation_consistency():
if True:
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
    rotator=ants.contrib.Rotate3D(rotation=(degrotx,degroty,degrotz), reference=img_LR_in_avg )
    rotation = rotator.transform()
    img_rotated_ref = ants.apply_ants_transform_to_image(rotation, img_LR_in_avg, reference=img_LR_in_avg)
    img_rotated = antspymm.timeseries_transform(rotation, img_LR_in, reference=img_LR_in_avg)
    mask_rotated = ants.apply_ants_transform_to_image(rotation, mask, reference=img_LR_in_avg, interpolation='nearestNeighbor')
    # note: we apply the inverse rotation to the bvecs
    # i.e. if we register A to B and get R then apply R_inv to bvecs
    rotmat = ants.get_ants_transform_parameters(rotation.invert()).reshape((4, 3))[:3, :3]
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
    # smoothing simulates double interpolation
    FA_origs = ants.smooth_image(FA_orig, 1.25)
    MD_origs = ants.smooth_image(MD_orig, 1.25)
    fa1 = extract_masked_values(FA_origs, maske)
    fa2 = extract_masked_values(FA_rot_back, maske)
    md1 = extract_masked_values(MD_origs, maske)
    md2 = extract_masked_values(MD_rot_back, maske)

    fa_corr, _ = pearsonr(fa1, fa2)
    md_corr, _ = pearsonr(md1, md2)

#    plot_correlation( fa1, fa2 )

    print(f"✅ FA correlation (original vs rotated): {fa_corr:.4f}")
    print(f"✅ MD correlation (original vs rotated): {md_corr:.4f}")
    
    assert fa_corr > 0.80, "FA correlation too low"
    assert md_corr > 0.80, "MD correlation too low"

    print("🎉 Test passed: model is rotation-consistent with voxelwise bvecs.")

    print("This shows that the simulation is effective.")
    print("Now use the simulated data to test the distortion correction...")
#    ants.image_write( RGB_orig, '/tmp/temp0rgb.nii.gz' )
#    ants.image_write( RGB_rot, '/tmp/temp1rgb.nii.gz' )


    # now we have to map the img_rotated and its bvec_rotated partner 
    # as we would with a generic distortion correction framework. 
    # first --- write the transform to a file
    rotationinv = rotation.invert()

    import tempfile
    with tempfile.TemporaryDirectory() as tempdir:
        rotmatfile = os.path.join(tempdir, "rotation.mat")
        compositefile = os.path.join(tempdir, "composite")

        # Write the rotation transform
        ants.write_transform(rotationinv, rotmatfile)

        # Run registration if not already done
        if "reg" not in globals():
            reg = ants.registration(FA_orig, FA_rot, 'SyN', initial_transform=rotmatfile)

        # Apply the transformation using a temporary composite path
        comptx = ants.apply_transforms(
            img_LR_in_avg,
            img_rotated_ref,
            reg['fwdtransforms'],
            interpolator='linear',
            compose=compositefile,
            verbose=True
        )
        mydef=ants.image_read(comptx)
        mydefgrad = antspymm.deformation_gradient_optimized( mydef, 
                to_rotation=False, to_inverse_rotation=True )
#        bvecsRLw = antspymm.generate_voxelwise_bvecs( bvecs_rotated, mydefgrad, transpose=False )
        img_rotated_avg = ants.get_average_of_timeseries(  img_rotated )
        bvecsRLw = antspymm.distortion_correct_bvecs( bvecs_rotated, mydefgrad, img_LR_in_avg.direction, img_rotated_avg.direction )
        mywarp = ants.transform_from_displacement_field( mydef )
        img_w = antspymm.timeseries_transform(mywarp, img_rotated, reference=img_rotated_avg )

        correlations = []
        labels = ["NoBvecReo", "BvecReo"]

        RGB_origs = ants.smooth_image(RGB_orig, 1.0)

        for label, bv in zip(labels, [bvecs_5d_rot, bvecsRLw]):
            FA_w, MD_w, RGB_w = antspymm.efficient_dwi_fit_voxelwise(
                imagein=img_w,
                maskin=ants.get_mask(ants.get_average_of_timeseries(img_w)),
                bvals=bvals,
                bvecs_5d=bv,
                model_params={},
                bvals_to_use=None,
                num_threads=nt,
                verbose=False
            )
            # Optional: write RGB images to temporary files for inspection
            rgb_orig_path = os.path.join(tempdir, "RGB_orig.nii.gz")
            rgb_warped_path = os.path.join(tempdir, f"RGB_warped_{label}.nii.gz")

            ants.image_write(RGB_origs, rgb_orig_path)
            ants.image_write(RGB_w, rgb_warped_path)

            print(f"📝 Saved RGB_orig to: {rgb_orig_path}")
            print(f"📝 Saved RGB_warped ({label}) to: {rgb_warped_path}")

            compmask = ants.get_mask(ants.get_average_of_timeseries(img_w))
            compmask = ants.iMath(compmask, 'ME', 2)

            corr = mean_rgb_correlation(RGB_origs, RGB_w, compmask)
            print(f"{label:10s} correlation: {corr:.4f}")
            correlations.append(corr)

# Compare the two correlations
if len(correlations) == 2:
    print("\nComparison Result:")
    if correlations[1] > correlations[0]:
        print("✅ BvecReo gives higher correlation than NoBvecReo.")
    else:
        print("❌ NoBvecReo gives equal or higher correlation than BvecReo.")