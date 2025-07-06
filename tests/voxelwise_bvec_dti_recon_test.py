import numpy as np
from scipy.spatial.transform import Rotation as R
import antspymm

def test_generate_voxelwise_bvecs():
    # Define synthetic bvecs (N = 3 directions)
    bvecs = np.array([
        [1, 0, 0],   # x-axis
        [0, 1, 0],   # y-axis
        [0, 0, 1],   # z-axis
    ])
    
    # Define image shape (small synthetic 4D volume)
    X, Y, Z = 2, 2, 2
    N = bvecs.shape[0]
    
    # Define a known rotation (e.g., 90-degree rotation around z-axis)
    rot = R.from_euler('z', 33, degrees=True).as_matrix()  # shape (3, 3)
    
    # Create a voxelwise rotation field (X, Y, Z, 3, 3), same rot everywhere
    voxel_rotations = np.tile(rot, (X, Y, Z, 1, 1))
    voxel_rotations = voxel_rotations.reshape(X, Y, Z, 3, 3)
    
    # Expected rotated bvecs
    expected = np.dot(voxel_rotations[0, 0, 0], bvecs.T).T  # (N, 3)

    # Call the function under test
    bvecs_5d = antspymm.generate_voxelwise_bvecs(bvecs, voxel_rotations)  # shape (X, Y, Z, N, 3)
    
    # Check that all voxel outputs match expected result
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                actual = bvecs_5d[i, j, k]
                assert np.allclose(actual, expected, atol=1e-6), \
                    f"Mismatch at voxel {(i, j, k)}: {actual} vs {expected}"

    print("✅ test_generate_voxelwise_bvecs passed!")

# Run the test
print("Running test for generate_voxelwise_bvecs...")
test_generate_voxelwise_bvecs()


import numpy as np
import ants
from scipy.spatial.transform import Rotation as R

def generate_dummy_dwi_data(shape, n_volumes):
    """
    Generate synthetic DWI-like 4D data and a brain mask.
    """
    np.random.seed(42)
    dwi = np.random.rand(*shape, n_volumes).astype(np.float32)
    mask = np.ones(shape, dtype=np.uint8)
    return dwi, mask

def create_voxelwise_bvecs(shape, bvecs, rotation_matrix=None):
    """
    Create voxelwise (5D) bvecs array (X, Y, Z, N, 3), optionally rotated.
    """
    if rotation_matrix is not None:
        rotated_bvecs = (rotation_matrix @ bvecs.T).T
    else:
        rotated_bvecs = bvecs

    # Broadcast to 5D shape
    bvecs_5d = np.broadcast_to(rotated_bvecs, shape + rotated_bvecs.shape)
    return bvecs_5d.copy()

def test_efficient_dwi_fit_voxelwise():

    # Parameters
    shape = (3, 3, 3)
    n_vols = 6

    # Synthetic data
    dwi_data, mask_data = generate_dummy_dwi_data(shape, n_vols)
    ants_dwi = ants.from_numpy(dwi_data)
    ants_mask = ants.from_numpy(mask_data)

    # Define bvals and bvecs
    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000])
    bvecs = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(2), 0, 1/np.sqrt(2)],
    ])

    # Optional: apply rotation
    rotation_matrix = R.from_euler('z', 45, degrees=True).as_matrix()
    bvecs_5d = create_voxelwise_bvecs(shape, bvecs, rotation_matrix)

    # Call function under test
    FA_img, MD_img, RGB_img = antspymm.efficient_dwi_fit_voxelwise(
        imagein=ants_dwi,
        maskin=ants_mask,
        bvals=bvals,
        bvecs_5d=bvecs_5d,
        model_params={},
        bvals_to_use=None,
        num_threads=1,
        verbose=False
    )

    # Tests
    assert isinstance(FA_img, ants.ANTsImage), "FA_img should be an ANTsImage"
    assert FA_img.shape == shape, f"FA image has shape {FA_img.shape}, expected {shape}"
    assert np.all((FA_img.numpy() >= 0) & (FA_img.numpy() <= 1)), "FA values should be in [0, 1]"
    assert MD_img.shape == shape, "MD image has incorrect shape"
    
    print("✅ test_efficient_dwi_fit_voxelwise passed!")

# Run the test
test_efficient_dwi_fit_voxelwise()
