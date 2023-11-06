import ants
import pandas as pd
import numpy as np

def create_segmentation_figures(statistical_file_path, data_dictionary_path, output_prefix, edge_image_path, edge_image_dilation = 0, verbose=False ):
    """
    Create segmentation figures based on statistical data and an edge image.

    Parameters:
    - statistical_file_path (str): Path to the statistical data CSV file.
    - data_dictionary_path (str): Path to the data dictionary CSV file.
    - output_prefix (str): Prefix for the output figure filenames.
    - edge_image_path (str): Path to the edge image in NIfTI format.
    - edge_image_dilation: integer greater than or equal to zero
    - verbose: boolean

    Returns:
    None
    """

    # Read the statistical file
    zz = pd.read_csv(statistical_file_path)
    ocols = zz.keys()
    zz.rename(columns={ocols[0]: 'X'}, inplace=True)
    zz['XX'] = zz['X'].str.replace(r'(vol_|thk_|LRAVG_|_LRAVG|Asym_|volAsym|volLRAVG|thkAsym|thkLRAVG)', '', regex=True)

    # Read the data dictionary from a CSV file
    mydict = pd.read_csv(data_dictionary_path)
    mydict = mydict[~mydict['Measurement'].str.contains("tractography-based connectivity", na=False)]

    # Load image and process it
    edgeimg = ants.image_read(edge_image_path)
    if edge_image_dilation > 0:
        edgeimg = ants.iMath( edgeimg, "MD", edge_image_dilation)

    # Define lists and data frames
    postfix = ['bf', 'deep_cit168lab', 'mtl', 'cerebellum', 'dkt_cortex']
    atlas = ['BF', 'CIT168', 'MTL', 'TustisonCobra', 'desikan-killiany-tourville']
    postdesc = ['nbm3CH13', 'CIT168_Reinf_Learn_v1_label_descriptions_pad', 'mtl_description', 'cerebellum', 'dkt']
    statdf = pd.DataFrame({'img': postfix, 'atlas': atlas, 'csvdescript': postdesc})

    # Iterate through columns and create figures
    for col2viz in zz.columns[1:4]:
        anattoshow = zz['XX'].unique()
        if verbose:
            print(col2viz)
            print(anattoshow)
        # Rest of your code for figure creation goes here...
        addem = edgeimg * 0
        for k in range(len(anattoshow)):
            vals2viz = zz[zz['XX'].str.contains(anattoshow[k])][col2viz].agg(['min', 'max'])
            vals2viz = vals2viz[abs(vals2viz).idxmax()]
            myext = None
            if 'dktcortex' in anattoshow[k]:
                myext = 'dkt_cortex'
            elif 'cit168' in anattoshow[k]:
                myext = 'deep_cit168lab'
            elif 'mtl' in anattoshow[k]:
                myext = 'mtl'
            elif 'cerebellum' in anattoshow[k]:
                myext = 'cerebellum'
            elif any(item in anattoshow[k] for item in ['nbm', 'bf']):
                myext = 'bf'
            for j in postfix:
                if j == "dkt_cortex":
                    j = 'dktcortex'
                if j == "deep_cit168lab":
                    j = 'deep_cit168'
                anattoshow[k] = anattoshow[k].replace(j, "")
            if verbose:
                print( anattoshow[k] + " " + str( vals2viz ) )
            myatlas = atlas[postfix.index(myext)]
            correctdescript = postdesc[postfix.index(myext)]
            locfilename = prefix + myext + '.nii.gz'
            myatlas = ants.image_read(locfilename)
            atlasDescript = pd.read_csv(f"~/.antspyt1w/{correctdescript}.csv")
            atlasDescript['Description'] = atlasDescript['Description'].str.lower()
            atlasDescript['Description'] = atlasDescript['Description'].str.replace(" ", "_")
            atlasDescript['Description'] = atlasDescript['Description'].str.replace("_left_", "_")
            atlasDescript['Description'] = atlasDescript['Description'].str.replace("_right_", "_")
            atlasDescript['Description'] = atlasDescript['Description'].str.replace("_left", "")
            atlasDescript['Description'] = atlasDescript['Description'].str.replace("_right", "")
            if myext == 'cerebellum':
                atlasDescript['Description'] = atlasDescript['Description'].str.replace("l_", "")
                atlasDescript['Description'] = atlasDescript['Description'].str.replace("r_", "")
                whichindex = atlasDescript.index[atlasDescript['Description'] == anattoshow[k]].values[0]
            else:
                whichindex = atlasDescript.index[atlasDescript['Description'].str.contains(anattoshow[k])]

            if type(whichindex) is np.int64:
                labelnums = atlasDescript.loc[whichindex, 'Label']
            else:
                labelnums = list(atlasDescript.loc[whichindex, 'Label'])
            if not isinstance(labelnums, list):
                labelnums=[labelnums]
            addemiszero = ants.threshold_image(addem, 0, 0)
            temp = ants.image_read(locfilename)
            temp = ants.mask_image(temp, temp, level=labelnums, binarize=True)
            temp[temp == 1] = abs(vals2viz)
            temp[addemiszero == 0] = 0
            addem = addem + temp

        if verbose:
            print('Done Adding')
        for axx in range(3):
            figfn=output_prefix+f"fig{col2viz}ax{axx}_py.jpg"
            cmask = ants.threshold_image( edgeimg,0.001, 1e9 ).iMath("MD",10)
            addemC = ants.crop_image( addem, cmask )
            edgeimgC = ants.crop_image( edgeimg, cmask )
            ants.plot(edgeimgC, addemC, axis=axx, nslices=42, ncol=7,       
                overlay_cmap='turbo', resample=False,
                vminol=0.01, vmaxol=addem.max(),
                filename=figfn, cbar=True, crop=True  )
        if verbose:
            print(f"{col2viz} done")
    if verbose:
        print("DONEzo")


# Example file paths and output prefix
statistical_file_path = "/tmp/temp.csv"
data_dictionary_path = "~/code/ANTsPyMM/docs/antspymm_data_dictionary.csv"
output_prefix = '/tmp/vizit_'
edge_image_path = '~/PPMI_template0_edge.nii.gz'

# Call the function
create_segmentation_figures(statistical_file_path, data_dictionary_path, output_prefix, edge_image_path, 1, True )

