import ants
import pandas as pd
import numpy as np
import antspymm


# Example file paths and output prefix
statistical_file_path = '/Users/stnava/code/multidisorder/caches/asyn_subtyping_idiopathic_PD_ppmi_k4_rank_0_nsimlr_0_pdpos_seed2_balancemwmote_msc2/asyn_subtyping_idiopathic_PD_ppmi_k4_rank_0_nsimlr_0_pdpos_seed2_balancemwmote_msc2_st_PD.csv'
statistical_file_path="/Users/stnava/code/multidisorder/caches/asyn_subtyping_idiopathic_PD_ppmi_k4_rank_0_nsimlr_0_pdpos_seed3_balancemwmote_msc2/asyn_subtyping_idiopathic_PD_ppmi_k4_rank_0_nsimlr_0_pdpos_seed3_balancemwmote_msc2_st_AR.csv"
data_dictionary_path = "~/code/ANTsPyMM/docs/antspymm_data_dictionary.csv"
output_prefix = '/tmp/vizit_'
edge_image_path = '~/.antspymm/PPMI_template0_edge.nii.gz'
edge_image_path = '~/.antspymm/PPMI_template0_brain.nii.gz'
brain_image = ants.image_read( edge_image_path )
brain_image_t = ants.iMath( brain_image, 'TruncateIntensity', 0.002, 0.99)
# Call the function
zz = pd.read_csv( statistical_file_path )
ocols = zz.keys()
zz.rename(columns={ocols[0]: 'subtype'}, inplace=True)

for myco in zz['subtype']:
    print( myco + " begin" )
    temp = zz[zz["subtype"].isin([myco])]
    temp = temp.set_index( 'subtype')
    zzz = temp.transpose().reset_index()
    zzz.rename(columns={'index': 'anat'}, inplace=True)
    zzz.rename(columns={myco: 'value'}, inplace=True)
    qqq = zzz.copy()
    qqq['anat'] = qqq['anat'].str.replace(r'(vol_|thk_|LRAVG_|_LRAVG|Asym_|_Asym|volAsym|volLRAVG|thkAsym|thkLRAVG)', '', regex=True)
    olimg = antspymm.brainmap_figure(qqq, data_dictionary_path, output_prefix + myco, brain_image_t, nslices=21, black_bg=False, axes=[1], fixed_overlay_range=[-1.0,1.0],verbose=True )
    print( myco + ' done')
