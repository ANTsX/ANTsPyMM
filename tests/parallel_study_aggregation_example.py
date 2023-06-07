import antspymm
import glob as glob
import re
import pandas as pd
import os
import numpy as np

import sys

# check if an argument was passed
if len(sys.argv) == 2:
    try:
        # convert the parameter to an integer
        num = int(sys.argv[1])
        print("The integer entered is:", num)
    except ValueError:
        print("Invalid input. Please enter an integer.")
else:
    print("Please enter an integer as a command line input parameter.")
    num=1


df = pd.read_csv( "matched_mm_data2.csv" )
pdir='./processedCSV/'
df['projectID']='PPMI'

outfn="split/ppmi_matched_qc_mm_"+"chunk_"+str(num)+".csv"
print(outfn)
k = 10
chunks = np.array_split(df, k)

merged = antspymm.merge_wides_to_study_dataframe( chunks[num], pdir, verbose=False, report_missing=False, progress=20 )
print(merged.shape)
merged.to_csv(outfn)

