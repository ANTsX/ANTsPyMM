import sys, os
import unittest

os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import tempfile
import shutil
import tensorflow as tf
import antspymm
import antspyt1w
import antspynet
import ants
import numpy as np
from scipy.stats import median_abs_deviation
import math
testingClass = unittest.TestCase( )
islocal = False
idp = "LS2001_3T_rfMRI_REST1_LR_gdc"
fmri = ants.image_read( antspymm.get_data( idp, target_extension=".nii.gz") )
fmri_template, hlinds = antspymm.loop_timeseries_censoring( fmri, 0.1 )