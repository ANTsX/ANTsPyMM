##################################################################
# convert to pynb via p2j mm.py -o
# convert the ipynb to html via:
#   jupyter nbconvert ANTsPyMM/tests/mm.ipynb --execute --to html
#
# this assumes NRG format for the input data .... 
# we also assume that t1w hierarchical is already done.
# NRG = https://github.com/stnava/biomedicalDataOrganization
##################################################################
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
import re
import nibabel as nib
import numpy as np
import ants
import antspymm
import antspyt1w
import pandas as pd
import glob
import antspynet
from os.path import exists
###########################################################
import tensorflow as tf
import tensorflow.keras as keras
# srmdl = tf.keras.models.load_model( '~/code/DPR/models/dsr3d_2up_64_256_6_3_v0.0.h5', compile=False )


antspymm.mm_nrg( 
    sourcedir = os.path.expanduser( "~/data/PPMI/MV/example_s3_b/images/PPMI/" ),
    sid  = "100898",   # subject unique id
    dtid = "20210331", # date
    iid  = "1496183",  # image unique id for t1 - should have highest grade if repeats exist
    sourcedatafoldername = 'images',
    processDir = "processed",
    mysep = '-', # define a separator for filename components
    srmodel = None,
    visualize = True,
    verbose=True
)
