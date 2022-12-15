##################################################################
# convert to pynb via p2j mm_nrg.py -o
# convert the ipynb to html via:
#   jupyter nbconvert ANTsPyMM/tests/mm_nrg.ipynb --execute --to html
#
# this assumes NRG format for the input data ....
# NRG = https://github.com/stnava/biomedicalDataOrganization
##################################################################
import os
from os.path import exists
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
mydir = os.path.expanduser( "~/PPMI500/source/data/PPMI/" )
srOption = False
# set srOption True to run auto-selected SR ... must call antspymm.get_data() first.
testfn = os.path.expanduser( "~/.antspymm/siq_default_sisr_2x2x2_1chan_featvggL6_best_mdl.h5" )
if not exists( testfn ):
    print("downloading sr models - takes a few GB of space")
    antspymm.get_data()
else:
    print("SR models are here ... " + testfn )
antspymm.mm_nrg(
        sourcedir = mydir,
        sid  = "100267",   # subject unique id
        dtid = "20210219", # date
        iid  = "1497590",  # image unique id for t1 - should have highest grade if repeats exist
        sourcedatafoldername = 'source',
        processDir = "processed",
        mysep = '-', # define a separator for filename components
        srmodel_T1 = srOption,
        srmodel_NM = srOption,
        srmodel_DTI = srOption,
        visualize = True,
        verbose=True
    )
