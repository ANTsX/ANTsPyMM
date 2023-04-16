#!/usr/bin/env python3
"""\

unzip ukbb t1 data -  map to nrg format - viz - qc - process

"""

import ants
import antspymm
import pandas as pd
import glob as glob
import os
from os.path import exists
import re
import math
from dipy.io.gradients import read_bvals_bvecs
def getnrgid( fn2org ):
    temp=os.path.basename(fn2org).split("-")
    print(temp)
    projectID=temp[0]
    subjectID=temp[1]
    modalityID=re.sub(".nii.gz","",temp[4])
    visitID=temp[2]
    modder=temp[3]
    return projectID, subjectID, visitID, modder, modalityID

def tonrg( fn2org, modder="T1w" ):
    fn2org=str(fn2org)
    if modder == "DTI" or modder == "T1w" or modder == "T2Flair" or modder == "rsfMRI" :
        temp=os.path.basename(fn2org).split("_")
        print(temp)
        if temp[0]=='nan':
            return None
        print(fn2org)
        print(modder)
        projectID='UKBB'
        subjectID=temp[0]
        modalityID=temp[1]
        visitID=temp[2]
        print(projectID + " : " + subjectID + " " + visitID + " " + modder + " " + modalityID )
        import shutil
        extract_dir=os.path.join( "/mnt/cluster/data/UKBB/nrg/", projectID, subjectID, visitID, modder, modalityID )
        print( extract_dir )
        if not os.path.isdir( extract_dir ):
            os.makedirs( extract_dir, exist_ok=True  )
            shutil.unpack_archive( fn2org, extract_dir)
        uid=projectID + "-" + subjectID + "-" + visitID + "-" + modder + "-" + modalityID
        idim=3
        if modder == "T1w":
            readfn="/T1/T1.nii.gz"
        elif modder == "T2Flair":
            readfn="/T2_FLAIR/T2_FLAIR.nii.gz"
        elif modder == "rsfMRI":
            readfn="/fMRI/rfMRI.nii.gz"
            idim=4
        elif modder == "DTI":
            readfn="/dMRI/dMRI/data_ud.nii.gz"
            idim=4
        else:
            raise ValueError("not doable")
        img=ants.image_read( extract_dir + readfn )
        ofn=extract_dir + "/" + uid + ".nii.gz"
        if img.dimension == idim:
            ants.image_write( img, ofn )
        else:
            raise ValueError( uid + " bad dimension: " + str(img.dimension) )
        print(uid, " done ")
        if modder == "DTI":
            print("dobv")
            bvalfn=extract_dir+"/dMRI/dMRI/bvals"
            bvecfn=extract_dir+"/dMRI/dMRI/data.eddy_rotated_bvecs"
            bvals, bvecs = read_bvals_bvecs( bvalfn, bvecfn )
            antspymm.write_bvals_bvecs( bvals, bvecs, re.sub(".nii.gz","",ofn) )
        return ofn
    else:
        print("not done")

index=int( os.environ["SLURM_ARRAY_TASK_ID"] )
print(index)
mm=pd.read_csv( "/mnt/cluster/data/UKBB/modality_map.csv" )
allzips=pd.read_csv("/mnt/cluster/data/UKBB/current_ukbb_zips.csv")
subviz=allzips[ 'T1w_fullfn' ].iloc[index]
print( allzips.iloc[index] )
nrgt1fn = tonrg( subviz )
print( nrgt1fn )
# same for T2 
nrgt2fn = tonrg( allzips[ 'T2Flair_fullfn' ].iloc[index], "T2Flair" )
print( nrgt2fn )

# same for rsfMRI
nrgrsfn = tonrg( allzips[ 'rsfMRI_fullfn' ].iloc[index], "rsfMRI" )
print( nrgrsfn )

# same for DTI 
nrgdtfn0 = tonrg( allzips[ 'DTI_fullfn' ].iloc[index], "DTI" )
print( nrgdtfn0 )

projectID, subjectID, date, modder, imageID = getnrgid( nrgt1fn )

studycsv = antspymm.generate_mm_dataframe(
    projectID,
    subjectID,
    date,
    imageID, # the T1 id
    'T1w',
    './nrg/',
    './processed/',
    t1_filename=nrgt1fn,
    flair_filename=[nrgt2fn],
    dti_filenames=[nrgdtfn0],
    rsf_filenames=[nrgrsfn])
studycsv2 = studycsv.dropna(axis=1)

qcdir = os.path.dirname( nrgt1fn ) + "/QC/"
os.makedirs( qcdir+"viz", exist_ok=True  )

antspymm.quick_viz_mm_nrg(
    './nrg/', # root for source data
    projectID,
    subjectID , # subject unique id
    date, # date
    extract_brain=True,
    slice_factor = 0.55,
    show_it = qcdir+"viz/viz_it",
    verbose = True )

# bqa
myresam=2.0
vizfn=None
mypr=False
qct1=antspymm.blind_image_assessment( nrgt1fn, vizfn, title=True, resample=myresam, pull_rank=mypr, verbose=True )
qclist=[qct1]
fnlist=[ nrgt2fn, nrgdtfn0, nrgrsfn]
for fn in fnlist:
    if fn is not None:
        locqc = antspymm.blind_image_assessment( fn, vizfn, title=True, resample=myresam, pull_rank=mypr, verbose=True )
        qclist.append( locqc )
allqc=pd.concat( qclist, axis=0 )
allqc.to_csv( qcdir + "qc.csv" )
qcdfa=antspymm.average_blind_qc_by_modality(allqc,verbose=True) ## reduce the time series qc
qcdfa.to_csv( qcdir + "qc-average.csv" )
mmrun = antspymm.mm_csv( studycsv2, dti_motion_correct='Rigid', dti_denoise=False )
print( "mm done" )

