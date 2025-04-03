# powers points 10.1016/j.conb.2012.12.009
powers=read.csv("~/.antspymm/powers_mni_itk.csv")
# 10.1016/j.neuroimage.2008.07.009
# https://doi.org/10.1016/j.neuroimage.2007.07.053
jhu=read.csv("~/.antspyt1w/FA_JHU_labels_edited.csv")
# doi: 10.3389/fnins.2012.00171
# http://dx.doi.org/10.1016/j.neuroimage.2006.01.021
dktcsv=read.csv("~/.antspyt1w/dkt.csv")
dktcsv=dktcsv[dktcsv$Label>0,]
# hipp https://doi.org/10.1101/2023.01.17.23284693
hipp=read.csv("~/.antspyt1w/mtl_description.csv")
hipp$Anatomy=hipp$Description
hipp$Anatomy=gsub("alEC"," antero-lateral entorhinal cortex",hipp$Anatomy)
hipp$Anatomy=gsub("pMEC"," postero-medial entorhinal cortex",hipp$Anatomy)
hipp$Anatomy=gsub("DG"," dentate gyrus",hipp$Anatomy)
hipp$Anatomy=gsub("CA"," cornu ammonis", hipp$Anatomy)

# https://doi.org/10.1101/211201
cit=read.csv("~/.antspyt1w/CIT168_Reinf_Learn_v1_label_descriptions_pad.csv")
cit$Anatomy=NA
cit$Anatomy[  grep("STR_Ca", cit$Description )] = 'caudate'
cit$Anatomy[  grep("STR_Pu", cit$Description )] = 'putamen'
cit$Anatomy[  grep("STR_NAC", cit$Description )] = 'Nucleus Accumbens'
cit$Anatomy[  grep("VTA", cit$Description )] = 'Ventral Tegmental Area'
cit$Anatomy[  grep("PBP", cit$Description )] = 'Parabrachial Pigmented Nucleus'
cit$Anatomy[  grep("SNc", cit$Description )] = 'Substantia Nigra pars compacta'
cit$Anatomy[  grep("SNr", cit$Description )] = 'Substantia Nigra pars reticulated'
cit$Anatomy[  grep("GPe", cit$Description )] = 'globus pallidus externa'
cit$Anatomy[  grep("GPi", cit$Description )] = 'globus pallidus interna'
cit$Anatomy[  grep("RN", cit$Description )] = 'red nucleus'
cit$Anatomy[  grep("STH", cit$Description )] = 'Subthalamic Nucleus'
cit$Anatomy[  grep("HTH", cit$Description )] = 'Hypothalamus'
cit$Anatomy[  grep("HN", cit$Description )] = 'Habenular Nuclei'
cit$Anatomy[  grep("EXA", cit$Description )] = 'extended amygdala'
cit$Anatomy[  grep("BNST", cit$Description )] = 'bed nuclei of the stria terminali'
cit$Anatomy[  grep("MN", cit$Description )] = 'mammillary nucleus'
cit$Anatomy[  grep("SLEA", cit$Description )] = 'sublenticular extended amygdala'
cit$Anatomy[  grep("VeP", cit$Description )] = 'ventral pallidum'

interpretcnx<-function( x ) {
    breaker=gsub("DTI_cnxcount","",x)
    temp = unlist(strsplit(breaker,"_"))
    ind=temp[1]
    anat=paste( temp[-1],collapse='_')
    return( paste( anat, "to", dktcsv[as.integer(ind),'Description'] ) )
}
interpretcnx2<-function( x ) {
    breaker=gsub("DTI_cnxcount","",x)
    temp = unlist(strsplit(breaker,"_"))
    ind=temp[1]
    anat=paste( temp[-1],collapse='_')
    return( dktcsv[as.integer(ind),'Description'] )
}
# dd=read.csv("joined_mm_or2.csv")
zz=data.frame( Label=colnames(dd))
qcrows=min(grep("RandBasis",zz$Label)):grep("resnetGrade", zz$Label)
zz$Modality='Other'
zz[ grep("T1Hier", zz$Label), 'Modality']='T1 hierarchical processing'
zz[ grep("T1w", zz$Label), 'Modality']='T1 DiReCT thickness processing'
zz[ grep("DTI", zz$Label), 'Modality']='DTI'
zz[ grep("NM2DMT", zz$Label), 'Modality']='Neuromelanin'
zz[ grep("rsfMRI", zz$Label), 'Modality']='restingStatefMRI'
zz[ grep("lair", zz$Label), 'Modality']='Flair'
zz[ grep("left", zz$Label), 'side']='left'
zz[ grep("right", zz$Label), 'side']='right'
zz$Atlas='ANTs'
zz[ grep("dkt", zz$Label), 'Atlas']='desikan-killiany-tourville'
zz[ grep("cnxcou", zz$Label), 'Atlas']='desikan-killiany-tourville'
zz[ grep("jhu", zz$Label), 'Atlas']='johns hopkins white matter'
zz[ grep("cit", zz$Label), 'Atlas']='CIT168'
zz[ grep("nbm", zz$Label), 'Atlas']='BF'
zz[ grep("ch13", zz$Label), 'Atlas']='BF'
zz[ grep("mtl", zz$Label), 'Atlas']='MTL'
zz[ grep("rsfMRI", zz$Label),'Atlas']='power peterson fMRI meta-analyses'
zz[qcrows,'Atlas']='quality control metrics'
zz[qcrows,'Measurement']='QC'
zz$Measurement[  grep("FD", zz$Label)]='motion statistic on framewise displacement'
zz$Measurement[  grep("thk", zz$Label)]='geometry/thickness'
zz$Measurement[  grep("area", zz$Label)]='geometry/area'
zz$Measurement[  grep("vol", zz$Label)]='geometry/volume'
zz$Measurement[  grep("mean_md", zz$Label)]='mean diffusion'
zz$Measurement[  grep("mean_fa", zz$Label)]='fractional anisotropy'
zz$Measurement[  grep("cnx", zz$Label)]='tractography-based connectivity'
zz$Anatomy = zz$Label
zz$Anatomy = gsub("_thk_","", zz$Anatomy)
zz$Anatomy = gsub("_area_","", zz$Anatomy)
zz$Anatomy = gsub("_volume_","", zz$Anatomy)
zz$Anatomy = gsub("DTI_cnxcount","", zz$Anatomy)
zz$Anatomy = gsub("DTI_mean_md","", zz$Anatomy)
zz$Anatomy = gsub("DTI_mean_fa","", zz$Anatomy)
zz$Anatomy = gsub("T1Hier_","", zz$Anatomy)
zz$Anatomy = gsub("T1Hier","", zz$Anatomy)
# fix dkt
dktlabs=dktcsv$Description
dktlabs=gsub("right ","",dktlabs)
dktlabs=gsub("left ","",dktlabs)
dktlabs2=gsub(" ","_",dktlabs)
for ( k in 1:length(dktlabs) ) {
    gg=grep( dktlabs[k], zz$Label)
    zz[ gg, "Atlas"]="desikan-killiany-tourville"
    zz[ gg, "Anatomy"]=dktlabs[k]
    gg=grep( dktlabs2[k], zz$Label)
    zz[ gg, "Atlas"]="desikan-killiany-tourville"
    zz[ gg, "Anatomy"]=dktlabs[k]
}

# fix cit
citlabs=tolower( cit$Description)
for ( k in 1:length(citlabs) ) {
    gg=grep( citlabs[k], zz$Label)
    zz[ gg, "Atlas"]="CIT168"
    zz[ gg, "Anatomy"]=cit$Anatomy[k]
}
zz$Anatomy = gsub("DTIfa","", zz$Anatomy)
zz$Anatomy = gsub("DTImd","", zz$Anatomy)
zz$Anatomy = gsub("dktregions","", zz$Anatomy)
zz$Anatomy = gsub("dktcortex"," cortex only ", zz$Anatomy)
zz$Anatomy = gsub("_right_","", zz$Anatomy)
zz$Anatomy = gsub("_left_","", zz$Anatomy)
zz$Anatomy = gsub("right","", zz$Anatomy)
zz$Anatomy = gsub("left","", zz$Anatomy)
zz$Anatomy = gsub("jhu_icbm_labels_1mm","", zz$Anatomy)
zz[ grep("u_hier_id", zz$Label), -1 ]='unique id'
cnxrows=grep("DTI_cnxcount",zz$Label)
for ( k in cnxrows )
    zz$Anatomy[k]=interpretcnx( zz[k,'Label'] )

zz[ multigrep( c("rsfMRI","R"), zz$Label, intersect=TRUE), 'side'  ]='right'
zz[ multigrep( c("rsfMRI","L"), zz$Label, intersect=TRUE), 'side'  ]='left'
zz$Measurement[ multigrep( c("rsfMRI","_2_"), zz$Label, intersect=TRUE) ]='network correlation'
zz$Measurement[ multigrep(c("rsfMRI","_alff"), zz$Label, intersect=TRUE) ]='amplitude of low frequency fluctuations ALFF'
zz$Measurement[ multigrep( c("rsfMRI","_falff"), zz$Label, intersect=TRUE) ]='fractional amplitude of low frequency fluctuations fALFF'
zz$Anatomy = gsub("rsfMRI_", "", zz$Anatomy )
zz$Anatomy = gsub("falffPoint", "", zz$Anatomy )
zz$Anatomy = gsub("alffPoint", "", zz$Anatomy )
noncnx=1:1888
for ( k in sample(noncnx, 3) ) print( zz[k,c("Label","Atlas","Anatomy")] )

zz[ zz$Label == 'Flair', 'Measurement' ]='white matter hyper-intensity'
zz[ zz$Label == 'T2Flair_flair_wmh_prior', 'Measurement' ]='prior-constrained white matter hyper-intensity'

zz[ multigrep( c("NM2DMT", "q0pt"),  zz$Label, intersect=TRUE), "Measurement"  ]='neuromelanin intensity quantile'

write.csv( zz, "~/code/ANTsPyMM/antspymm_data_dictionary.csv", row.names=FALSE)
