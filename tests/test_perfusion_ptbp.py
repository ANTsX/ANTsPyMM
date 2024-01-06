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
t1fn = "/tmp/data/PEDS015/20110812/Anatomy/PEDS015_20110812_mprage_t1.nii.gz"
idpfn = "/tmp/data/PEDS015/20110812/PCASL/PEDS015_20110812_pcasl_1.nii.gz"
t1fn = '/tmp/data/PEDS022/20111216/Anatomy/PEDS022_20111216_mprage_t1.nii.gz'
idpfn = '/tmp/data/PEDS022/20111216/PCASL/PEDS022_20111216_pcasl_1.nii.gz'
t1fn = '/tmp/data/PEDS041/20110127/Anatomy/PEDS041_20110127_mprage_t1.nii.gz'
idpfn = '/tmp/data/PEDS041/20110127/PCASL/PEDS041_20110127_pcasl_1.nii.gz'
t1fn='/tmp/data/PEDS014/20100831/Anatomy/PEDS014_20100831_mprage_t1.nii.gz'
idpfn='/tmp/data/PEDS014/20100831/PCASL/PEDS014_20100831_pcasl_1.nii.gz'
t1fn = "/Users/stnava/data/PTBP/PTBP/PEDS049/20110217/T1w/000/PTBP-PEDS049-20110217-T1w-000.nii.gz"
idpfn = "/Users/stnava/data/PTBP/PTBP/PEDS049/20110217/perf/000/PTBP-PEDS049-20110217-perf-000.nii.gz"
t1fn = "/Users/stnava/data/PTBP/PTBP/PEDS147/20131220/T1w/000/PTBP-PEDS147-20131220-T1w-000.nii.gz"
idpfn = "/Users/stnava/data/PTBP/PTBP/PEDS147/20131220/perf/000/PTBP-PEDS147-20131220-perf-000.nii.gz"
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
fmri = ants.image_read( idpfn )
fmri_template, hlinds = antspymm.loop_timeseries_censoring( fmri, 0.1 )
fmri_template = ants.get_average_of_timeseries( fmri_template )
print("do perf")
# olthresh=0.5
perf = antspymm.bold_perfusion( fmri, fmri_template, t1head, t1, 
  t1segmentation, dkt, type_of_transform=type_of_transform, verbose=True )
ants.image_write( ants.iMath( perf['perfusion'], "Normalize" ), '/tmp/temp.nii.gz' )
ants.image_write( perf['motion_corrected'], '/tmp/temp2.nii.gz' )
ants.image_write( perf['cbf'], '/tmp/temp3ptb.nii.gz' )
ants.plot( ants.iMath( perf['perfusion'], "Normalize" ), axis=2, crop=True )
ants.plot( ants.iMath( perf['cbf'], "Normalize" ), axis=2, crop=True )

if False:
    print("do perf2")
    perf2 = antspymm.bold_perfusion( fmri, fmri_template, t1head, t1, 
    t1segmentation, dkt, nc=16, type_of_transform=type_of_transform,
    spa=(0.,0.,0.,0.),
    outlier_threshold=olthresh, 
    add_FD_to_nuisance=False,
    segment_timeseries=True,
    verbose=True )
    ants.plot( ants.iMath( perf2['perfusion'], "Normalize" ), axis=2, crop=True )
