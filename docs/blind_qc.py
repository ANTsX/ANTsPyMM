##################################################################
import os
import sys
from os.path import exists
nthreads = str(2)
index=0
if len( sys.argv ) > 1 :
    index=int( sys.argv[1] )
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
import random as random
from os.path import exists
# where the NRG data is stored
mydir = os.path.expanduser( "/mnt/cluster/data/MyStudy/" )
# get all the subject t1 images
if not exists( "images_to_qc.csv" ):
    print("Collect all images" )
    mymods = antspymm.get_valid_modalities( )
    afns0=[]
    for m in mymods:
        afns0.append( glob.glob( mydir + "*/*/*/*/*"+m+"*nii.gz" ) )
    afns = []
    afns = [item for sublist in afns0 for item in sublist]
    afns.sort()
    df = pd.DataFrame(afns, columns=['filename'])
    df.to_csv( "images_to_qc.csv" )
    
df = pd.read_csv( "images_to_qc.csv" )
from pathlib import Path
mypr=False
myresam=None # 2.0
# mypr=True
odir='vizx'
off=21
indexLo=index*off
indexHi=(index+1)*off
for index2 in range( indexLo, indexHi ):
    if index2 < df.shape[0]:
        ifn=df['filename'].iloc[index2]
        mystem=Path( ifn ).stem    
        mystem=Path( mystem ).stem    
        vizfn=odir+'/viz_'+mystem+'.png'
        csvfn=odir+'/viz_'+mystem+'.csv'
        if not exists( csvfn ):
            print("index: " + str(index) + " threads " + nthreads + " " + vizfn)
            try:
                antspymm.blind_image_assessment( ifn, vizfn, title=True,
                        resample=myresam, pull_rank=mypr, verbose=True )
            except:
                pass

