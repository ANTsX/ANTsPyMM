
fns = Sys.glob("./processed//PPMI/*/*/T1wHierarchical/*/*-mmwide.csv")
admissiblemods=c("T1w",
    "T2Flair","rsfMRI","rsfMRI_LR","rsfMRI_RL","DTI","DTI_LR","DTI_RL")
for ( k in 1:length( fns ) ) {
    startdf = read.csv( fns[k] )
    rootcolnames = colnames( startdf )
    temp = fns[k]
    mypartsf = strsplit( temp, "T1wHierarchical" )
    myparts = mypartsf[[1]][1]
    # handle nm
    fnsnm = Sys.glob(paste0(myparts,"/NM2DMT/*wide.csv" ))
    if ( length( fnsnm ) == 1 ) {
      nmcsv = read.csv( fnsnm )[1,]
      startdf = cbind( startdf, nmcsv[,-1] )
    }
    for ( j in 1:length(admissiblemods) ) {
        fnsnm = Sys.glob(paste0(myparts,"/", admissiblemods[j], "/*/*wide.csv" ))
        if ( length( fnsnm ) == 1 ) {
            dd = read.csv( fnsnm )
            cnxcoutnames = grep("cnxcount",colnames(dd))
            if ( length( cnxcoutnames ) > 0 )
              dd = dd[ , -cnxcoutnames ]
            inames = intersect( rootcolnames, colnames(dd) )
            dd = dd[ , !(colnames(dd) %in% inames)  ]
            tagger = paste0( admissiblemods[j],"_")
            if ( length(dim(dd)) > 0 ) {
                if ( dim(dd)[1] == 2 ) {
                    dd=dd[2,]
                    grepinner = c( grep("inner",dd[1,]), grep("outer",dd[1,]) )
                    if ( length( grepinner ) > 0  )
                        dd=dd[,-grepinner]
                    }
                colnames(dd)=paste0(tagger,colnames(dd))
                startdf = cbind( startdf, dd )
            }
#            print(admissiblemods[j])
 #           print( dim(startdf ))
            }
        }
    print("final")
    write.csv(startdf,'thename.csv') # FIXME decide on name
    }
