import antspymm


## TEST 
bids_filename = 'sub-10874_ses-I450998_func.nii.gz'
date = '20210127'
project_name = 'PPMI'
nrg = antspymm.bids_2_nrg(bids_filename, project_name, date)
print("bids_filename: " + bids_filename )
print("converted nrg: " + nrg )


## TEST 
nrg_filename = 'PPMI-10874-20140905-rsfMRI-I450998.nii.gz'
bids = antspymm.nrg_2_bids(nrg_filename)
print("nrg_filename: " + nrg_filename )
print("converted bids: " + bids )

