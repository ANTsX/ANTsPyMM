##################################################################
# convert to pynb via p2j mm_nrg.py -o
# convert the ipynb to html via:
#   jupyter nbconvert ANTsPyMM/tests/mm_nrg.ipynb --execute --to html
#
# this assumes NRG format for the input data .... 
# NRG = https://github.com/stnava/biomedicalDataOrganization
##################################################################
import os
nthreads = str(8)
os.environ["TF_NUM_INTEROP_THREADS"] = nthreads
os.environ["TF_NUM_INTRAOP_THREADS"] = nthreads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
import antspymm
###########################################################
import tensorflow as tf
import tensorflow.keras as keras
mdlfn = os.path.expanduser( '~/code/DPR/models/dsr3d_2up_64_256_6_3_v0.0.h5' )
srmdl = tf.keras.models.load_model( mdlfn, compile=False )
doOr = True
doSr = False
doviz = True
mydir = os.path.expanduser( "~/data/PPMI/MV/example_s3_b/images/PPMI/" )
if doOr:
    antspymm.mm_nrg(
        sourcedir = mydir,
        sid  = "100898",   # subject unique id
        dtid = "20210331", # date
        iid  = "1496183",  # image unique id for t1 - should have highest grade if repeats exist
        sourcedatafoldername = 'images',
        processDir = "processedOR",
        mysep = '-', # define a separator for filename components
        srmodel_NM = None,
        srmodel_DTI = None,
        visualize = doviz,
        verbose=True
    )

if doSr:
    print("SR processing - will be slow")
    antspymm.mm_nrg(
        sourcedir = mydir,
        sid  = "100898",   # subject unique id
        dtid = "20210331", # date
        iid  = "1496183",  # image unique id for t1 - should have highest grade if repeats exist
        sourcedatafoldername = 'images',
        processDir = "processedSR",
        mysep = '-', # define a separator for filename components
        srmodel_NM = srmdl,
        srmodel_DTI = srmdl,
        visualize = doviz,
        verbose=True
    )
