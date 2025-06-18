##################################################################
# convert to pynb via p2j mm_nrg.py -o
# convert the ipynb to html via:
#   jupyter nbconvert ANTsPyMM/tests/mm_nrg.ipynb --execute --to html
#
# this assumes NRG format for the input data ....
# NRG = https://github.com/stnava/biomedicalDataOrganization
##################################################################
import os
import sys
from os.path import exists
nthreads = str(8)
index=0
if len( sys.argv ) > 1 :
    index=int( sys.argv[1] )
    nthreads=str(sys.argv[2])
os.environ["TF_NUM_INTEROP_THREADS"] = nthreads
os.environ["TF_NUM_INTRAOP_THREADS"] = nthreads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
import antspymm
import sys
import glob as glob
import os
import pandas as pd
import re as re
mydir = os.path.expanduser( "/mnt/cluster/data/PPMI500/source/data/PPMI/" )
print("index: " + str(index) + " threads " + nthreads )
# get all the subject t1 images
t1fns = glob.glob( mydir + "*/*/T1w/*/*T1w*nii.gz" )
df = antspymm.nrg_filelist_to_dataframe( t1fns, '-' )
#################
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
        sid  = df['sid'].iloc[index],   # subject unique id
        dtid = df['visitdate'].iloc[index], # date
        iid  = df['uid'].iloc[index],  # image unique id for t1 - should have highest grade if repeats exist
        sourcedatafoldername = 'source',
        processDir = "processed",
        mysep = '-', # define a separator for filename components
        srmodel_T1 = srOption,
        srmodel_NM = srOption,
        srmodel_DTI = srOption,
        visualize = True,
        verbose=True
    )
