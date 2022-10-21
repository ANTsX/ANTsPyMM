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
ex_path = os.path.expanduser( "~/.antspyt1w/" )
templatefn = ex_path + 'CIT168_T1w_700um_pad_adni.nii.gz'
if not exists( templatefn ):
    print( "**missing files** => call get_data from latest antspyt1w and antspymm." )
    stophere
template = ants.image_read( templatefn ) # Read in template
###########################################################
import tensorflow as tf
import tensorflow.keras as keras
# srmdl = tf.keras.models.load_model( '~/code/DPR/models/dsr3d_2up_64_256_6_3_v0.0.h5', compile=False )
mysep = '-' # define a separator for filename components
subjectrootpath = os.path.expanduser( "~/data/PPMI/MV/example_s3_b/images/PPMI/100007/20201209/" )
myimgs = glob.glob( subjectrootpath+"*" )
# hierarchical
# NOTE: if there are multiple T1s for this time point, should take 
# the one with the highest resnetGrade
t1fn = glob.glob( subjectrootpath + "/T1w/1525725/*nii.gz")[0]
t1 = ants.image_read( t1fn )
hierexists = True # FIXME should test this explicitly but we assume it here
hierfn = os.path.expanduser( "~/data/PPMI/MV/example_s3_b/processed/PPMI/100007/20201209/T1wHierarchical/1525725/100007-20201209-T1wHierarchical-1525725-")
hier = antspyt1w.read_hierarchical( hierfn )
t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(
    hier['dataframes'], identifier=None )
t1imgbrn = hier['brain_n4_dnz']
t1atropos = hier['dkt_parc']['tissue_segmentation']
ants.plot( t1imgbrn,  axis=2, nslices=21, ncol=7, crop=True, title='brain extraction' )
ants.plot( t1imgbrn, t1atropos, axis=2, nslices=21, ncol=7, crop=True, title='segmentation'  )
ants.plot( t1imgbrn, hier['dkt_parc']['dkt_cortex'], axis=2, nslices=21, ncol=7, crop=True, title='cortex'   )
# loop over modalities and then unique image IDs
# we treat NM in a "special" way -- aggregating repeats 
# other modalities (beyond T1) are treated individually
for x in myimgs[1:4]:
    myimgsr = glob.glob( x+"/*" )
    overmod = x.split( "/" )
    overmod = overmod[ len(overmod)-1 ]
    if overmod == 'NM2DMT':
        myimgsr = glob.glob( x+"/*/*nii.gz" )
        subjectpropath = re.sub( 'images', 'processed', x ) + "MM"
        mysplit = subjectpropath.split( "/" )
        os.makedirs( subjectpropath, exist_ok=True  )
        identifier = mysplit[9] + mysep + mysplit[10] + mysep + 'NM2DMTMM' + mysep
        mymm = subjectpropath + "/" + identifier
        nmlist = []
        for zz in myimgsr:
            nmlist.append( ants.image_read( zz ) )
        tabPro, normPro = antspymm.mm( t1, hier, 
                    nm_image_list = nmlist,
                    srmodel=None,
                    do_tractography=False, 
                    do_kk=False, 
                    do_normalization=True, 
                    verbose=True )
        antspymm.write_mm( output_prefix=mymm, mm=tabPro, mm_norm=normPro, t1wide=None, separator=mysep )
        nmpro = tabPro['NM']
        mysl = range( nmpro['NM_avg'].shape[2] )
        ants.plot( nmpro['NM_avg'],  nmpro['t1_to_NM'], slices=mysl, axis=2, title='nm + t1' )
        mysl = range( nmpro['NM_avg_cropped'].shape[2] )
        ants.plot( nmpro['NM_avg_cropped'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop' )
        ants.plot( nmpro['NM_avg_cropped'], nmpro['t1_to_NM'], axis=2, slices=mysl, overlay_alpha=0.3, title='nm crop + t1' )
        ants.plot( nmpro['NM_avg_cropped'], nmpro['NM_labels'], axis=2, slices=mysl, title='nm crop + labels' )
    else :
        for y in myimgsr:
            myimg = glob.glob( y+"/*nii.gz" )
            subjectpropath = re.sub( 'images', 'processed', y )
            mysplit = subjectpropath.split("/")
            mymod = mysplit[11] # FIXME system dependent
            uid = mysplit[12]
            if  mymod == 'T1w' and not hierexists : # what you would do to compute hier
                t1img = ants.image_read( myimg[0] )
                subjectpropath = re.sub( "T1w", "T1wHierarchical", subjectpropath )
                os.makedirs( subjectpropath, exist_ok=True  )
                mysplit = subjectpropath.split("/")
                mymod = mysplit[11] # FIXME system dependent
                identifier = mysplit[9] + mysep + mysplit[10] + mysep + mymod + mysep + uid + mysep
                myh = subjectpropath + "/" + identifier
                hier = antspyt1w.hierarchical( t1img, myh, labels_to_register=None )
                antspyt1w.write_hierarchical( hier, myh )
            else :
                print("Modality specific processing: " + mymod )
                mymodnew = mymod + "MM"
                subjectpropath = re.sub( mymod, mymodnew, subjectpropath )
                os.makedirs( subjectpropath, exist_ok=True  )
                identifier = mysplit[9] + mysep + mysplit[10] + mysep + mymodnew + mysep + uid
                mymm = subjectpropath + "/" + identifier
                print(subjectpropath)
                print(identifier)
                img = ants.image_read( myimg[0] )
                if mymod == 'T1' and False: # for a real run, set to True
                    tabPro, normPro = antspymm.mm( t1, hier, 
                        srmodel=None,
                        do_tractography=False, 
                        do_kk=True, 
                        do_normalization=True, 
                        verbose=True )
                    ants.plot( hier['brain_n4_dnz'], tabPro['kk']['thickness_image'], axis=2, nslices=21, ncol=7, crop=True, title='kk' )
                if mymod == 'T2Flair':
                    tabPro, normPro = antspymm.mm( t1, hier, 
                        flair_image = img,
                        srmodel=None,
                        do_tractography=False, 
                        do_kk=False, 
                        do_normalization=True, 
                        verbose=True )
                    ants.plot( img,   axis=2, nslices=21, ncol=7, crop=True, title='Flair' )
                    ants.plot( img, tabPro['flair']['WMH_probability_map'],  axis=2, nslices=21, ncol=7, crop=True, title='Flair + WMH' )
                if mymod == 'rsfMRI_LR' or mymod == 'rsfMRI_RL' :
                    tabPro, normPro = antspymm.mm( t1, hier, 
                        rsf_image=img,
                        srmodel=None,
                        do_tractography=False, 
                        do_kk=False, 
                        do_normalization=True, 
                        verbose=True )
                    ants.plot( tabPro['rsf']['meanBold'], tabPro['rsf']['DefaultMode'],
                        axis=2, nslices=21, ncol=7, crop=True, title='DefaultMode' )
                    ants.plot( tabPro['rsf']['meanBold'], tabPro['rsf']['FrontoparietalTaskControl'],
                        axis=2, nslices=21, ncol=7, crop=True, title='FrontoparietalTaskControl' )
                if mymod == 'DTI_LR' or mymod == 'DTI_RL' or mymod == 'DTI':
                    bvalfn = re.sub( '.nii.gz', '.bval' , myimg[0] )
                    bvecfn = re.sub( '.nii.gz', '.bvec' , myimg[0] )
                    tabPro, normPro = antspymm.mm( t1, hier, 
                        dw_image=img,
                        bvals = bvalfn,
                        bvecs = bvecfn,
                        srmodel=None,
                        do_tractography=False, 
                        do_kk=False, 
                        do_normalization=True, 
                        verbose=True )
                    mydti = tabPro['DTI']
                    ants.plot( mydti['dtrecon_LR']['FA'],  axis=2, nslices=21, ncol=7, crop=True, title='FA pre correction' )
                    ants.plot( mydti['recon_fa'],  axis=2, nslices=21, ncol=7, crop=True, title='FA (supposed to be better)' )
                    ants.plot( mydti['recon_fa'], mydti['jhu_labels'], axis=2, nslices=21, ncol=7, crop=True, title='FA + JHU' )
                    ants.plot( mydti['recon_md'],  axis=2, nslices=21, ncol=7, crop=True, title='MD' )
                antspymm.write_mm( output_prefix=mymm, mm=tabPro, mm_norm=normPro, t1wide=t1wide, separator=mysep )
                for mykey in normPro.keys():
                    if normPro[mykey] is not None:
                        ants.plot( template, normPro[mykey], axis=2, nslices=21, ncol=7, crop=True, title=mykey  )
