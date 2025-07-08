################################################################
#  for easier to access data with a full mm_csv example, see:  #
#  github.com:stnava/ANTPD_antspymm                            #
################################################################
import os
seed = 42  #
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# this is important for reading models via siq.read_srmodel
nthreads = str(48) # for much faster performance and good reproducibility
os.environ["TF_NUM_INTEROP_THREADS"] = nthreads
os.environ["TF_NUM_INTRAOP_THREADS"] = nthreads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
import random
import numpy as np
random.seed(seed)
np.random.seed(seed)
from os.path import exists
import signal
import urllib.request
import zipfile
import tempfile
from pathlib import Path
from tqdm import tqdm
import siq
import antspynet
print("read the SR model ")
mfn=os.path.expanduser('~/.antspymm/siq_smallshort_train_2x2x2_1chan_featgraderL6_best.keras')
mdl, mdlshape = siq.read_srmodel(mfn)
print("read the SR model done")
########################################################################
import numpy as np
import glob as glob
import antspymm
import ants
import antspyt1w
import random
import re
print("Begin template loading")
tlrfn = antspyt1w.get_data('T_template0_LR', target_extension='.nii.gz' )
tfn = antspyt1w.get_data('T_template0', target_extension='.nii.gz' )
templatea = ants.image_read( tfn )
templatea = ( templatea * antspynet.brain_extraction( templatea, 't1' ) ).iMath( "Normalize" )
templatealr = ants.image_read( tlrfn )
print("done template loading")

import antspymm #####
import pandas as pd #
import glob as glob #
rdir = os.path.expanduser( "~/Downloads/nrgdata_test/PPMI/" )
t1fn = glob.glob( rdir + "101018/20210412/T1w/1496225/*.nii.gz")
if len(t1fn) > 0:
    t1fn=t1fn[0]
    testimg = ants.image_read( t1fn )
    bxt  = antspyt1w.brain_extraction( testimg )
    imgb = testimg*bxt
    mylr = antspyt1w.label_hemispheres( imgb, templatea, templatealr )
    mysr = siq.inference( testimg, mdl, segmentation=mylr*bxt, truncation=[0.001,0.999], poly_order='hist', verbose=True )
    print("done SR -- overwrite the T1w image " + t1fn + " with super_resolution")
    ants.image_write( ants.iMath( mysr, "Normalize"),  '/tmp/tempsr.nii.gz' )
    
    