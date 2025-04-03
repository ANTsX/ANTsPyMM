
######## template figures ########
import ants
import antspymm
import pandas as pd
# Example Usage
scalar_label_df = pd.DataFrame({'label': range(33), 'scalar_value': range(33)})
prefix = '../PPMI_template0_'
print("begin")
for t in ['mtl','bf','jhuwm','cerebellum','cit168']:
    print( t )
    proimgs=antspymm.template_figure_with_overlay(scalar_label_df, prefix, template=t, outputfilename='/tmp/py_viz_'+t+'.png' )
t='ctx'
scalar_label_df = pd.DataFrame({'label': range(1001,1200), 'scalar_value': range(1001,1200)})
proimgs=antspymm.template_figure_with_overlay(scalar_label_df, prefix, template=t, mask_dilation=0, outputfilename='/tmp/py_viz_'+t+'.png')


