import sys, os
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
t1fn = "./PEDS049/20110217/T1w/000/PTBP-PEDS049-20110217-T1w-000.nii.gz"
idpfn = "./PEDS049/20110217/perf/000/PTBP-PEDS049-20110217-perf-000.nii.gz"
imgp = ants.image_read( idpfn )
#dkt
if not 'dkt' in globals():
  t1head = ants.image_read( t1fn ).n3_bias_field_correction( 8 ).n3_bias_field_correction( 4 )
  t1bxt = antspynet.brain_extraction( t1head, 't1' ).threshold_image( 0.3, 1.0 )
  t1 = t1bxt * t1head
  t1seg = antspynet.deep_atropos( t1head )
  t1segmentation = t1seg['segmentation_image']
  dkt = antspynet.desikan_killiany_tourville_labeling( t1head )
#################
type_of_transform='Rigid'
tc='alternating'
fmri = ants.image_clone( imgp )
fmri_template = ants.get_average_of_timeseries( imgp )
print("do perf")
perf = antspymm.bold_perfusion( imgp, fmri_template, t1head, t1, 
  t1segmentation, dkt, nc=4,
  outlier_threshold=0.2, add_FD_to_nuisance=True, verbose=True )
# ants.plot( ants.iMath(perf['perfusion'],"Normalize"), axis=2, crop=True )
