### an example of production job submitted to a slurm q
import os
import sys
from os.path import exists
nthreads = str(8)
index=0
if len( sys.argv ) > 1 :
    index=int( sys.argv[1] )+int( sys.argv[3] )
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
mydir = os.path.expanduser( "/mnt/cluster/data/A4/nrg/A4/" )
df=pd.read_csv( "/mnt/cluster/data/A4/t1fns.csv" )
import antspymm
import pandas as pd
import glob as glob
t1fn=df['t1'][index]
temp=t1fn.split( "/" )
pid = temp[6]
myid = temp[7]
uid=temp[10]
mydate=temp[8]
flfn=glob.glob(mydir + myid + "/"+ mydate+ "/*/*/*T2Fla*nii*")
rsfn=glob.glob(mydir + myid + "/"+ mydate+ "/*/*/*rsfMRI*nii*")
studycsv = antspymm.generate_mm_dataframe(
    pid,
    myid,
    mydate,
    uid,
    'T1w',
    '/mnt/cluster/data/A4/nrg/',
    '/mnt/cluster/data/A4/processed/',
    t1fn,
    rsf_filenames=rsfn,
    flair_filename=flfn
)
studycsv2 = studycsv.dropna(axis=1)
print( studycsv2)
mmrun = antspymm.mm_csv( studycsv2, mysep='-' )

