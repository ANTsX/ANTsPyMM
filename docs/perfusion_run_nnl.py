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

def replicate_list(user_list, target_size):
    # Calculate the number of times the list should be replicated
    replication_factor = target_size // len(user_list)
    # Replicate the list and handle any remaining elements
    replicated_list = user_list * replication_factor
    remaining_elements = target_size % len(user_list)
    replicated_list += user_list[:remaining_elements]
    return replicated_list

def one_hot_encode(char_list):
    unique_chars = list(set(char_list))
    encoding_dict = {char: [1 if char == c else 0 for c in unique_chars] for char in unique_chars}
    encoded_matrix = np.array([encoding_dict[char] for char in char_list])
    return encoded_matrix

islocal = False
bdir=os.path.expanduser('~/.antspymm/')
t1fn = bdir + 'sub-1001-0462_run-001_t1.nii.gz'
ffn= bdir + 'sub-1001-0462_run-001_perf.nii.gz'
fmri = ants.image_read( ffn )
verbose=True
#dkt
t1head = ants.image_read( t1fn ).n3_bias_field_correction( 8 ).n3_bias_field_correction( 4 )
t1bxt = antspynet.brain_extraction( t1head, 't1' ).threshold_image( 0.3, 1.0 )
t1 = t1bxt * t1head

if not 't1seg' in globals():
  t1seg = antspynet.deep_atropos( t1head )

t1segmentation = t1seg['segmentation_image']

if not 'dkt' in globals():
  dkt = antspynet.desikan_killiany_tourville_labeling( t1head )
#################

#################
type_of_transform='Rigid'
tc='alternating'
fmri_template, hlinds = antspymm.loop_timeseries_censoring( fmri, 0.1 )
fmri_template = ants.get_average_of_timeseries( fmri_template )
print("do perf")
olthresh=0.2
perf = antspymm.bold_perfusion( 
  fmri, 
  t1head, 
  t1, 
  t1segmentation, dkt, nc=4, 
  type_of_transform=type_of_transform,
  spa=(0.,0.,0.,0.),
  outlier_threshold=olthresh, add_FD_to_nuisance=False, verbose=True )

# ants.image_write( ants.iMath( perf['perfusion'], "Normalize" ), '/tmp/temp.nii.gz' )
# ants.image_write( perf['motion_corrected'], '/tmp/temp2.nii.gz' )
ants.plot( ants.iMath( perf['perfusion'], "Normalize" ), axis=2, crop=True )
