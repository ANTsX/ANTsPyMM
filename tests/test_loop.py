import sys, os
import unittest
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
import tempfile
import shutil
import antspymm
import antspyt1w
import antspynet
import ants
import numpy as np
from scipy.stats import median_abs_deviation
import math
def test_loop_timeseries_censoring():
    idp = "LS2001_3T_rfMRI_REST1_LR_gdc"
    data_path = antspymm.get_data( idp, target_extension=".nii.gz")
    if not os.path.exists(data_path):
        return
    fmri = ants.image_read( data_path )
    fmri_template, hlinds1 = antspymm.loop_timeseries_censoring( fmri, 0.5, seed=0 )
    fmri_template2, hlinds2 = antspymm.loop_timeseries_censoring( fmri, 0.5, seed=1 )
    fmri_template3, hlinds3 = antspymm.loop_timeseries_censoring( fmri, 0.5, seed=2 )
    assert fmri_template is not None
    assert isinstance(hlinds1, (list, np.ndarray))
    assert isinstance(hlinds2, (list, np.ndarray))
    assert isinstance(hlinds3, (list, np.ndarray))
    assert len(hlinds1) > 0
