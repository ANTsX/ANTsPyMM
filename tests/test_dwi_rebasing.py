# based on https://github.com/cookpa/antsDTOrientationTests
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
from os.path import exists
from dipy.io.image import save_nifti, load_nifti
import antspymm
import antspyt1w
import antspynet
import ants
import pandas as pd
import tensorflow as tf
from tempfile import mktemp
import numpy as np
import antspymm
print(" Load in JHU atlas and labels ")
ex_path = os.path.expanduser( "~/.antspyt1w/" )
ex_path_mm = os.path.expanduser( "~/.antspymm/" )
JHU_atlas = ants.image_read( ex_path + 'JHU-ICBM-FA-1mm.nii.gz' ) # Read in JHU atlas
JHU_labels = ants.image_read( ex_path + 'JHU-ICBM-labels-1mm.nii.gz' ) # Read in JHU labels
#### Load in data ####
template = ants.image_read( "OASIS/T_template0_BrainCerebellum_3mm.nii.gz")
template = ants.resample_image(template,[2,2,2])
tmask = ants.get_mask( template ).iMath("MD",5)
template = ants.crop_image( template, tmask )
print("Load in subject data ...")
rotdir='yaw'
lrid='nii/'+rotdir+'/'+rotdir
img_LR_in = ants.image_read( lrid + '.nii.gz') # LR dwi image
img_LR_bval = lrid + '.bval' # bval
img_LR_bvec = lrid + '.bvec'

rotdir2='pitch'
rlid='nii/'+rotdir2+'/'+rotdir2
img_RL_in = ants.image_read( rlid + '.nii.gz') # LR dwi image
img_RL_bval = rlid + '.bval' # bval
img_RL_bvec = rlid + '.bvec'

print("build the DTI templates")
a1b,a1w=antspymm.get_average_dwi_b0(img_LR_in)

print("bxt the DTI template space")
mybxt = ants.get_mask( template )
reg = ants.registration( a1b, template, 'Rigid', verbose=False)
reg = ants.registration( a1b, template, 'SyNOnly', verbose=False, 
    initial_transform = reg['fwdtransforms'][0], total_sigma=5. )
mask = ants.apply_transforms( a1b, mybxt, reg['fwdtransforms'], interpolator='nearestNeighbor')
mask = ants.iMath(mask,"MD",1)
ants.plot(a1b,mask,axis=2)

a1brot = ants.apply_transforms( template, a1b, reg['invtransforms'] )
a1wrot = ants.apply_transforms( template, a1w, reg['invtransforms'] )
maskrot = ants.apply_transforms( template, mask, reg['invtransforms'] )


# the 1st test looks at reorienting the tensor after reconstruction
myoutx = antspymm.joint_dti_recon(
        img_LR_in,
        img_LR_bval,
        img_LR_bvec,
        jhu_atlas = JHU_atlas,
        jhu_labels = JHU_labels,
        reference_B0=a1b,
        reference_DWI=a1w,
        srmodel = None,
        motion_correct = 'Rigid',
        brain_mask = mask,
        denoise = True,
        verbose = True )

dti0 = antspymm.get_dti( a1b, myoutx['dtrecon_LR_dewarp']['tensormodel'], return_image=True )
dti1 = antspymm.get_dti( a1b, myoutx['dtrecon_LR_dewarp']['tensormodel'], return_image=False )


# now apply the transform to the template
# 1. transform the tensor components
dtsplit = dti0.split_channels()
dtiw = []
for k in range(len(dtsplit)):
    dtiw.append( ants.apply_transforms( template, dtsplit[k], reg['invtransforms'] ) )
ants.plot( template, dtiw[0], axis=1, crop=True )
dtiw=ants.merge_channels(dtiw)
# reorient them locally: compose and get reo image
comptx = ants.apply_transforms( template, template, reg['invtransforms'], 
                                compose='/tmp/XXX' )
locrot = ants.deformation_gradient( ants.image_read(comptx), 
    to_rotation = True, py_based=True )
# rebase them to new space
rebaser = np.dot( np.transpose( template.direction  ), a1b.direction )
dtiw2tensor = antspymm.triangular_to_tensor( dtiw )
it = np.ndindex( template.shape )
for i in it:
    # direction * dt * direction.transpose();
    mmm = dtiw2tensor[i]
    # transform rebase
    locrotx = np.reshape( locrot[i], [3,3] )
    mmm = np.dot( mmm, np.transpose( locrotx ) )
    mmm = np.dot( locrotx, mmm )
    # physical space rebase
    mmm = np.dot( mmm, np.transpose( rebaser ) )
    mmm = np.dot( rebaser, mmm )
    dtiw2tensor[i] = mmm

xxx=antspymm.dti_numpy_to_image( template, dtiw2tensor )
ants.image_write( xxx, '/tmp/dtiw.nii.gz' )

### just test rebase first
it = np.ndindex( template.shape )
for i in it:
    # direction * dt * direction.transpose();
    mmm = dtiw2tensor[i]
    mmm = np.dot( mmm, np.transpose( rebaser ) )
    mmm = np.dot( rebaser, mmm )
    dtiw2tensor[i] = mmm

xxx=antspymm.dti_numpy_to_image( template, dtiw2tensor )
ants.image_write( xxx, '/tmp/dtiw.nii.gz' )
ants.image_write( template, '/tmp/template.nii.gz' )
# these results look correct ... 


# the 2nd test looks at reorienting the tensor reconstruction itself
# ie evaluating whether the reconstruction is correct when performed 
# in a different space than the acquisition(s)
print("Begin joint DTI recon")
myoutx = antspymm.joint_dti_recon(
        img_LR_in,
        img_LR_bval,
        img_LR_bvec,
        img_RL=img_RL_in,
        bval_RL=img_RL_bval,
        bvec_RL=img_RL_bvec,
        jhu_atlas = JHU_atlas,
        jhu_labels = JHU_labels,
        reference_B0=a1brot,
        reference_DWI=a1wrot,
        srmodel = None,
        motion_correct = 'Rigid',
        brain_mask = maskrot,
        denoise = True,
        verbose = True )

if True:
    ants.image_write( myoutx['recon_fa'], '/tmp/temp1fa.nii.gz'  )
    ants.image_write( myoutx['recon_md'], '/tmp/temp1md.nii.gz'  )
    ants.image_write( myoutx['dwi_LR_dewarped'], '/tmp/temp1moco.nii.gz'  )
    ants.image_write( myoutx['dtrecon_LR_dewarp']['RGB'], '/tmp/temp1rgb.nii.gz'  )


# make the DTI - see 
# https://dipy.org/documentation/1.7.0/examples_built/07_reconstruction/reconst_dti/#sphx-glr-examples-built-07-reconstruction-reconst-dti-py
# By default, in DIPY, values are ordered as (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz)
# in ANTs - we have: [xx,xy,xz,yy,yz,zz]
reoind = np.array([0,1,3,2,4,5]) # arrays are faster than lists
import dipy.reconst.dti as dti
dtiut = dti.lower_triangular(myoutx['dtrecon_LR_dewarp']['tensormodel'].quadratic_form)
it = np.ndindex( a1b.shape )
for i in it: # convert to upper triangular
    dtiut[i] = dtiut[i][ reoind ] # do we care if this is doing extra work?
dtiAnts = ants.from_numpy(dtiut,has_components=True)
ants.copy_image_info( a1b, dtiAnts )
ants.image_write(dtiAnts,'/tmp/dti.nii.gz')
# copy these data into a tensor 
dtinp = np.zeros(a1b.shape + (3,3), dtype=float)  
dtix = np.zeros((3,3), dtype=float)  
it = np.ndindex( a1b.shape )
for i in it:
    dtivec = dtiut[i] # in ANTs - we have: [xx,xy,xz,yy,yz,zz]
    dtix[0,0]=dtivec[0]
    dtix[1,1]=dtivec[3] # 2 for LT
    dtix[2,2]=dtivec[5] 
    dtix[0,1]=dtix[1,0]=dtivec[1]
    dtix[0,2]=dtix[2,0]=dtivec[2] # 3 for LT
    dtix[1,2]=dtix[2,1]=dtivec[4]
    dtinp[i]=dtix


