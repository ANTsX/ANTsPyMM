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
import math
import glob

testingClass = unittest.TestCase( )
id = 'PEDS131'; dt = '20130816' # high motion - bad volumes
id = "PEDS074"; dt = "20110803" # high motion - bad volumes
# id = 'PEDS144'; dt = '20131212' # not so bad
# id = "PEDS107"; dt = "20130118" # challenging
prefix = "/Users/stnava/data/PTBP/images/" + id + "/" + dt 
bold = glob.glob( prefix + '/BOLD/' + id + "_" + dt + "*bold*nii.gz" )
t1w = glob.glob( prefix + '/Anatomy/' + id + "_" + dt + "*t1.nii.gz" )
rsfn = bold[0]
print( rsfn )
t1fn = t1w[0]
import pandas as pd
print("do t1")
if not "t1" in globals():
    t1 = ants.image_read( t1fn ).n3_bias_field_correction( 8 ).n3_bias_field_correction( 4 )
    t1bxt = antspynet.brain_extraction( t1, 't1' ).threshold_image( 0.3, 1.0 )
    t1seg = antspynet.deep_atropos( t1 )
    t1segmentation = t1seg['segmentation_image']

if not 'rsfpro' in globals():
  print("do rsf: default")
  img1 = ants.image_read( rsfn )
  fmri_template, hlinds = antspymm.loop_timeseries_censoring( img1, 0.1 )
  fmri_template = antspymm.get_average_rsf( fmri_template )
#  img1 = antspymm.remove_volumes_from_timeseries(img1, list(range(55,4000)))
  rsfpro = antspymm.resting_state_fmri_networks(
    img1,
    fmri_template, t1 * t1bxt, t1segmentation,
    verbose=True )
  mm = { 'rsf': rsfpro }
  antspymm.write_mm( '/tmp/RSF', mm )  
  ants.plot( rsfpro['fmri_template'], rsfpro[ rsfpro['dfnname']], crop=True, axis=2 )
  ants.plot( rsfpro['fmri_template'], rsfpro['alff'], crop=True, axis=2 )
  ants.plot( rsfpro['fmri_template'], rsfpro['falff'], crop=True, axis=2 )
  ants.plot( rsfpro['fmri_template'], rsfpro['PerAF'], crop=True, axis=2 )
