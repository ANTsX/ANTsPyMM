library(ANTsR)
library(subtyper)
# collect all the folders of the form 
# ADNI/027_S_4804/Axial_DTI/2013-07-22_10_22_19.0/I381890
fns=FIXME # Sys.glob("ADNI/*/*/*/*")
#######
for ( k in 1:length( fns ) ) {
    pid = 'ADNI'
    modality='DTI'
    fn=tools::file_path_sans_ext( fns[k] )
    uid = basename( fn )
    fn=gsub("ADNI/","",tools::file_path_sans_ext( fns[k], T))
    temp0 = unlist( strsplit( fn , '/') )
    sid = temp0[1]
    dt = gsub("-","",substr( temp0[3],0,10))
    dcm2cmd = paste( "dcm2niix -z y ", fns[k] )
    system(dcm2cmd)
    mynii=Sys.glob( paste0( fns[k], "/*nii.gz"))
    if ( length(mynii) > 0 ) {
        rdir = '/mnt/cluster/data/ADNI/nrg/ADNI'
        odir = paste0( c(rdir,sid,dt,modality,uid),collapse='/')
        dir.create( odir, showWarnings = FALSE, recursive = TRUE )
        myfn = paste0(c(pid,sid,dt,modality,uid),collapse='-')
        checkfns = c(
            mynii[1],
            gsub("nii.gz","json",mynii[1]),
            gsub("nii.gz","bval",mynii[1]),
            gsub("nii.gz","bvec",mynii[1]) )
        outfns=paste0(odir,'/',myfn,c('.nii.gz','.json',  '.bval','.bvec' ) )
        if ( all( file.exists( checkfns )  ) ) {
            for ( zz in 1:length(outfns) ) {
                if ( zz == 1 ) {
                    print("GOTIT")
                    print( outfns[zz] )
                }
                system( paste( "mv ", checkfns[zz], outfns[zz] ) )
            }
        }
    }
}

