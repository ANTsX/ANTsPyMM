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
print("do perf")
perfh = antspymm.bold_perfusion( fmri, t1head, t1, 
  t1segmentation, dkt, perfusion_regression_model='linear', verbose=True )
ants.plot( perfh['cbf'], axis=2, crop=True )
# do your own CBF calculation
mycbf = antspymm.calculate_CBF(
  Delta_M=perfh['perfusion'], M_0=perfh['m0'], mask=perfh['brainmask'],
  Lambda=0.9, T_1=0.67, Alpha=0.68, w=1.0, Tau=1.5)


mm = { 'perf': perfh }
antspymm.write_mm( '/tmp/PRF', mm )
