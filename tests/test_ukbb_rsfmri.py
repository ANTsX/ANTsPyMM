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

import pandas as pd
rsf = ants.image_read( "/tmp/UKBB-1839399-2-rsfMRI-20227.nii.gz" )
t1 = ants.image_read("/tmp/UKBB-1839399-2-T1wHierarchical-20252-brain_n4_dnz.nii.gz")
t1segmentation=ants.image_read("/tmp/UKBB-1839399-2-T1wHierarchical-20252-tissue_segmentation.nii.gz")
bt = antspymm.get_average_rsf(rsf)
rsfpro = antspymm.resting_state_fmri_networks(
  rsf,
  bt,
  t1, 
  t1segmentation, f=[0.03, 0.08], spa=1.5, spt=0.5, nc=6, verbose=True)
