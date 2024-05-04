##################################################################
import os
import sys
from os.path import exists
nthreads = str(2)
index=811
if len( sys.argv ) > 1 :
    index=int( sys.argv[1] )
    nthreads=sys.argv[2]
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
mydir = os.path.expanduser( "/mnt/cluster/data/PPMIBIG/nrgdata_2024_05_all/data/PPMI/" )
# print("index: " + str(index) + " threads " + nthreads )
# get all the subject t1 images
qcfn = "images_to_qc_2024_May.csv"
if not exists( qcfn ):
    print("Collect all images" )
    mymods = antspymm.get_valid_modalities( )
    afns0=[]
    for m in mymods:
        afns0.append( glob.glob( mydir + "*/*/*/*/*"+m+"*nii.gz" ) )
    afns = []
    afns = [item for sublist in afns0 for item in sublist]
    afns.sort()
    df = pd.DataFrame(afns, columns=['filename'])
    df.to_csv( qcfn )
    print("zillakong")
    
df = pd.read_csv( qcfn )
n=df.shape[0]
from pathlib import Path
mypr=False
myresam=None # 2.0
# mypr=True
odir='vizx_2024'
off=round( n / 1000 )
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
            print(ifn, "index: " + str(index) + " threads " + nthreads + " " + vizfn)
            antspymm.blind_image_assessment( ifn, vizfn, title=True,
                        resample=myresam, pull_rank=mypr, verbose=True )

print( str(index) + " finnish")

