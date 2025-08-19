#!/usr/bin/env Rscript

# Optional (you had it earlier, used for multigrep convenience)
suppressWarnings(suppressMessages({
  if (!requireNamespace("subtyper", quietly = TRUE)) {
    message("Note: 'subtyper' not found; defining a local multigrep().")
    multigrep <- function(patterns, x, intersect = FALSE, ...) {
      mats <- lapply(patterns, function(p) grepl(p, x, ...))
      if (intersect) {
        which(Reduce(`&`, mats))
      } else {
        which(Reduce(`|`, mats))
      }
    }
  } else {
    library(subtyper)
    library(ANTsR)
  }
}))

# --------- Helper: safe, case-insensitive gsub (no perl side effects) ----------
gsubi <- function(pattern, replacement, x) gsub(pattern, replacement, x, ignore.case = TRUE)

# --------- Load reference atlases (keep your originals) ------------------------
powers = read.csv("~/.antspymm/powers_mni_itk.csv")
jhu    = read.csv("~/.antspyt1w/FA_JHU_labels_edited.csv")
dktcsv = read.csv("~/.antspyt1w/dkt.csv")
dktcsv = dktcsv[dktcsv$Label > 0, ]

# Hippocampal subfields (expand)
hipp = read.csv("~/.antspyt1w/mtl_description.csv")
hipp$Anatomy = hipp$Description
hipp$Anatomy = gsub("alEC", " antero-lateral entorhinal cortex", hipp$Anatomy)
hipp$Anatomy = gsub("pMEC", " postero-medial entorhinal cortex", hipp$Anatomy)
hipp$Anatomy = gsub("DG", " dentate gyrus", hipp$Anatomy)
hipp$Anatomy = gsub("CA", " cornu ammonis", hipp$Anatomy)

# CIT168 — curated mapping
cit = read.csv("~/.antspyt1w/CIT168_Reinf_Learn_v1_label_descriptions_pad.csv")
cit$Anatomy = NA
cit$Anatomy[grep("STR_Ca",  cit$Description)] = "caudate"
cit$Anatomy[grep("STR_Pu",  cit$Description)] = "putamen"
cit$Anatomy[grep("STR_NAC", cit$Description)] = "Nucleus Accumbens"
cit$Anatomy[grep("VTA",     cit$Description)] = "Ventral Tegmental Area"
cit$Anatomy[grep("PBP",     cit$Description)] = "Parabrachial Pigmented Nucleus"
cit$Anatomy[grep("SNc",     cit$Description)] = "Substantia Nigra pars compacta"
cit$Anatomy[grep("SNr",     cit$Description)] = "Substantia Nigra pars reticulated"
cit$Anatomy[grep("GPe",     cit$Description)] = "globus pallidus externa"
cit$Anatomy[grep("GPi",     cit$Description)] = "globus pallidus interna"
cit$Anatomy[grep("RN",      cit$Description)] = "red nucleus"
cit$Anatomy[grep("STH",     cit$Description)] = "Subthalamic Nucleus"
cit$Anatomy[grep("HTH",     cit$Description)] = "Hypothalamus"
cit$Anatomy[grep("HN",      cit$Description)] = "Habenular Nuclei"
cit$Anatomy[grep("EXA",     cit$Description)] = "extended amygdala"
cit$Anatomy[grep("BNST",    cit$Description)] = "bed nuclei of the stria terminali"
cit$Anatomy[grep("MN",      cit$Description)] = "mammillary nucleus"
cit$Anatomy[grep("SLEA",    cit$Description)] = "sublenticular extended amygdala"
cit$Anatomy[grep("VeP",     cit$Description)] = "ventral pallidum"

# --------- Connectivity label interpreters (your originals) --------------------
interpretcnx <- function(x) {
  breaker = gsub("DTI_cnxcount", "", x)
  temp    = unlist(strsplit(breaker, "_"))
  ind     = temp[1]
  anat    = paste(temp[-1], collapse = "_")
  return(paste(anat, "to", dktcsv[as.integer(ind), "Description"]))
}
interpretcnx2 <- function(x) {
  breaker = gsub("DTI_cnxcount", "", x)
  temp    = unlist(strsplit(breaker, "_"))
  ind     = temp[1]
  return(dktcsv[as.integer(ind), "Description"])
}

# --------- Load ANTsPyMM example matrix and construct zz ----------------------
dd = read.csv("docs/example_antspymm_output.csv")
# get the names we want to decode
qcnames_raw=antspymm_qc_names()
# --- Step 1: Parse the raw string into a clean vector of names ---
# The logic handles the tricky 'rsf3_reflection_errpsnr' by pre-processing it.
qcnames_raw_fixed <- gsub("errpsnr", "err psnr", qcnames_raw)
qcnames_vec <- unlist(strsplit(qcnames_raw_fixed, "\\s+"))
qcnames_vec <- qcnames_vec[nzchar(qcnames_vec)]
# --- Step 2: Decode each name and assemble a human-readable version ---
human_readable_names <- sapply(qcnames_vec, function(raw_name) {
  decoded <- decode_antspymm_label(raw_name)
  
  parts <- c()
  
  # Add Modality if it's known
  if (decoded$modality != "Unknown") {
    parts <- c(parts, decoded$modality)
  }
  
  # Add Anatomy if it's not a generic placeholder
  if (decoded$anatomy != "Global" && !startsWith(decoded$anatomy, "Unknown Core")) {
    parts <- c(parts, decoded$anatomy)
  }
  
  # Add Measurement if it's known
  if (decoded$measurement != "Unknown") {
    parts <- c(parts, decoded$measurement)
  }
  
  # If no parts were found, use the original name as a fallback
  if (length(parts) == 0) {
    return(raw_name)
  }
  
  # Combine parts into a single string
  final_name <- paste(parts, collapse = " ")
  
  # Add laterality if present
  if (decoded$laterality != "None") {
    final_name <- paste0(final_name, " (", decoded$laterality, ")")
  }
  
  return(final_name)
}, USE.NAMES = FALSE)


# --- Step 3: Create the final named vector ---
qc_results <- qcnames_vec
names(qc_results) <- human_readable_names
names(qc_results)[1]='AI T1 grading'
names(qc_results)[grepl("msk_vol",qc_results)]='Mask Volume'
ee=antspymm_predictors(dd)
idpnames=unique(colnames(ee))# [ !colnames(ee) %in% colnames(dd)]
names(idpnames)=idpnames
idpdf=data.frame(Label=idpnames, Description=NA, Modality=NA, Measurement=NA, stringsAsFactors=FALSE)
rownames(idpdf)=idpnames
for ( x in 1:length(idpnames) ) { 
  notfn = ! ( grepl("fn1",idpnames[x]) | grepl("fn2",idpnames[x]) | grepl("id1",idpnames[x]) | grepl("id2",idpnames[x]) )
  if ( !is.na( antspymm_vartype( idpnames[x] ) ) & notfn ) {
    mydec=decode_antspymm_idp( idpnames[x] )
    idpdf[idpnames[x], "Description"] = mydec$anatomy
    idpdf[idpnames[x], "Modality"] = mydec$modality
    idpdf[idpnames[x], "Measurement"] = mydec$measurement
  }
}

nuis=antspymm_nuisance_names()
names(nuis)=c("signal to noise", "bandpass filter", "mean of image", "censoring schema", "smoothing amount", "outlier amount", "motion related", "framewise displacement", "despiking count", "number of compcor components", "eigenvalue ratio", "minutes", 'laterality.l', 'laterality.r', 'parameter set', 'standard deviation', 'upsampling amount', 'mahalanobis distance','random basis projection', 'template L1 distance')

nuisdf=data.frame(Label=nuis, Description=names(nuis), stringsAsFactors=FALSE)
qcdf=data.frame(Label=qc_results, Description=names(qc_results), stringsAsFactors=FALSE)

# make all frames have the same columns
nuisdf$Modality <- NA
nuisdf$Measurement <- NA

qcdf$Modality <- NA
qcdf$Measurement <- NA

# bind them together
zz <- rbind(
  idpdf[, c("Label", "Description", "Modality", "Measurement")],
  nuisdf[, c("Label", "Description", "Modality", "Measurement")],
  qcdf[, c("Label", "Description", "Modality", "Measurement")]
)

# rownames(zz) <- zz$Label


zz[grep("T1Hier",  zz$Label), "Modality"] = "T1 hierarchical processing"
zz[grep("T1w",     zz$Label), "Modality"] = "T1 DiReCT thickness processing"
zz[grep("DTI",     zz$Label), "Modality"] = "DTI"
zz[grep("NM2DMT",  zz$Label), "Modality"] = "Neuromelanin"
zz[grep("rsfMRI",  zz$Label), "Modality"] = "restingStatefMRI"
zz[grep("lair|flair", zz$Label, ignore.case = TRUE), "Modality"] = "Flair"

zz$Atlas = "ANTs"
zz[grep("dkt",     zz$Label, ignore.case = TRUE), "Atlas"] = "desikan-killiany-tourville"
zz[grep("cnxcou",  zz$Label, ignore.case = TRUE), "Atlas"] = "desikan-killiany-tourville"
zz[grep("jhu",     zz$Label, ignore.case = TRUE), "Atlas"] = "johns hopkins white matter"
zz[grep("cit",     zz$Label, ignore.case = TRUE), "Atlas"] = "CIT168"
zz[grep("nbm",     zz$Label, ignore.case = TRUE), "Atlas"] = "BF"
zz[grep("ch13",    zz$Label, ignore.case = TRUE), "Atlas"] = "BF"
zz[grep("mtl",     zz$Label, ignore.case = TRUE), "Atlas"] = "MTL"
zz[grep("rsfMRI",  zz$Label, ignore.case = TRUE), "Atlas"] = "power peterson fMRI meta-analyses"

zz[grep("FD",      zz$Label, ignore.case = TRUE), "Measurement"] = "motion statistic on framewise displacement"
zz[grep("thk",     zz$Label, ignore.case = TRUE), "Measurement"] = "geometry.thickness"
zz[grep("area",    zz$Label, ignore.case = TRUE), "Measurement"] = "geometry.area"
zz[grep("vol",     zz$Label, ignore.case = TRUE), "Measurement"] = "geometry.volume"
zz[grep("mean_md", zz$Label, ignore.case = TRUE), "Measurement"] = "mean diffusion"
zz[grep("mean_fa", zz$Label, ignore.case = TRUE), "Measurement"] = "fractional anisotropy"
# zz[grep("cnx",     zz$Label, ignore.case = TRUE), "Measurement"] = "tractography-based connectivity"


# Unique id rows
if (length(grep("u_hier_id", zz$Label))) {
  zz[grep("u_hier_id", zz$Label), setdiff(colnames(zz), "Label")] = "unique id"
}

# Interpret DTI connectivity rows (yours)
cnxrows = grep("DTI_cnxcount", zz$Label, ignore.case = TRUE)
for (k in cnxrows) zz$Anatomy[k] = interpretcnx(zz[k, "Label"])



# --------- Yeo-17 + dopaminergic/limbic/striatum decoding for rsfMRI ----------
# Canonical network token map (supports A/B/C and 1/2/3)
.yeo_map <- c(
  "Vis1"="visual network 1", "Vis2"="visual network 2",
  "SomMotA"="somatomotor A", "SomMotB"="somatomotor B",
  "SomMot1"="somatomotor 1", "SomMot2"="somatomotor 2",
  "DorsAttnA"="dorsal attention A", "DorsAttnB"="dorsal attention B",
  "DorsAttn1"="dorsal attention 1", "DorsAttn2"="dorsal attention 2",
  "SalVentAttnA"="salience / ventral attention A", "SalVentAttnB"="salience / ventral attention B",
  "LimbicA"="limbic A (orbitofrontal)", "LimbicB"="limbic B (temporal pole)",
  "Limbic1"="limbic 1 (orbitofrontal)", "Limbic2"="limbic 2 (temporal pole)",
  "ContA"="frontoparietal control A", "ContB"="frontoparietal control B", "ContC"="frontoparietal control C",
  "Cont1"="frontoparietal control 1", "Cont2"="frontoparietal control 2", "Cont3"="frontoparietal control 3",
  "DefaultA"="default mode A", "DefaultB"="default mode B", "DefaultC"="default mode C",
  "Default1"="default mode 1", "Default2"="default mode 2", "Default3"="default mode 3",
  # extensions you requested
  "Dopa"="dopaminergic network", "Striatum"="striatal network", "Limbic"="limbic network"
)

# decode fcnxpro number (rsfMRI processing set)
extract_fcnxpro <- function(x) {
  m <- regmatches(x, regexpr("fcnxpro[0-9]+", x, ignore.case = TRUE))
  ifelse(length(m) == 0, NA, gsubi("fcnxpro", "", m))
}

# turn e.g. fcnxpro134_DorsAttnA_2_SomMotB into "dorsal attention A to somatomotor B"
decode_yeo_pair <- function(x) {
  # Find a pair around _2_
  m <- regexec("([A-Za-z0-9]+)_2_([A-Za-z0-9]+)", x)
  r <- regmatches(x, m)
  if (length(r) == 0 || length(r[[1]]) < 3) return(NA)
  a <- r[[1]][2]; b <- r[[1]][3]
  # Map via canonical dictionary; fall back to raw token if missing
  A <- .yeo_map[match(a, names(.yeo_map))]; if (is.na(A)) A <- a
  B <- .yeo_map[match(b, names(.yeo_map))]; if (is.na(B)) B <- b
  # Human-readable
  paste0(tolower(A), " to ", tolower(B))
}

# Apply only on rsfMRI rows; keep your Anatomy when not applicable
is_rsfmri <- grepl("rsfMRI", zz$Label, ignore.case = TRUE)
zz$rsf.pipeline   = NA
zz$rsf.pipeline[is_rsfmri] = sapply(zz$Label[is_rsfmri], extract_fcnxpro)

decoded_pairs <- rep(NA_character_, nrow(zz))
decoded_pairs[is_rsfmri] = sapply(zz$Label[is_rsfmri], decode_yeo_pair)

# If we successfully decode, replace the Anatomy with the human-readable pair
rep_idx <- which(is_rsfmri & !is.na(decoded_pairs))
if (length(rep_idx)) {
  zz$Anatomy[rep_idx] = decoded_pairs[rep_idx]
}

# --------- Domain-specific acronym polishing in Anatomy -----------------------
# Examples you requested: Pfclleft => prefrontal cortex; Exstr => extra-striatal; etc.
# We do this *after* major atlas mappings so we don’t clobber DKT/CIT names.
acro_map <- list(
  "(^|_)pfcl([_^]|$)"   = " prefrontal cortex ",
  "(^|_)exstr([_^]|$)"  = " extra-striatal ",
  "(^|_)nac([_^]|$)"    = " nucleus accumbens ",
  "(^|_)pu([_^]|$)"     = " putamen ",
  "(^|_)ca([_^]|$)"     = " caudate ",
  "(^|_)gp([_^]|$)"     = " globus pallidus ",
  "(^|_)amy([_^]|$)"    = " amygdala ",
  "(^|_)hpc([_^]|$)"    = " hippocampus ",
  "(^|_)pfclleft([_^]|$)" = " prefrontal cortex ",
  "(^|_)pfclright([_^]|$)"= " prefrontal cortex "
)
for (pat in names(acro_map)) {
#  zz$Anatomy <- gsub(pat, acro_map[[pat]], zz$Anatomy, ignore.case = TRUE, perl = TRUE)
}

# Whitespace tidy
# zz$Anatomy <- gsub("\\s+", " ", zz$Anatomy)
# zz$Anatomy <- trimws(zz$Anatomy)

# --------- Example prints (your sampling) -------------------------------------
noncnx = 1:min(1888, nrow(zz))
set.seed(1)
if (length(noncnx) >= 3) {
  for (k in sample(noncnx, 3)) print(zz[k, c("Label","Atlas","Anatomy")])
}

# --------- Special cases you had ---------------------------------------------
zz[zz$Label == "Flair", "Measurement"] = "white matter hyper-intensity"
zz[zz$Label == "T2Flair_flair_wmh_prior", "Measurement"] = "prior-constrained white matter hyper-intensity"
zz[multigrep(c("NM2DMT","q0pt"), zz$Label, intersect = TRUE), "Measurement"] = "neuromelanin intensity quantile"

clean_anatomy <- function(x) {
  if (is.null(x)) return(NA)
  x <- tolower(x)
  x <- str_remove_all(x, "fcnxpro[0-9]+_[0-9]+_")
  
  replacements <- c(
    "tempocc"="temporo-occipital",
    "temppole"="temporal pole",
    "pfcv"="ventral prefrontal cortex",
    "temppar"="temporo-parietal",
    "r_i_iicerebellum"="cerebellum (subdivision)",
    "spc"="spacing",
    "org"="origin",
    "sar"=" specific absorption ratio",
    "nbmmid"="nbm mid partition",
    "nbmant"="nbm anterior partition",
    "nbmpos"="nbm posterior partition",
    "dimt"="time dimension",
    "dim"="dimension",
    "noise"="imaging noise",
    "nm2dmt"="neuromelanin",
#    "nm"="neuromelanin",
#    "mi"="mutual information",
#    "ol"="outlier estimate",
    "upsampling"="upsampling rate",
    "ch13_bf"="ch13 basal forebrain",
    "ch4_bf"="ch4 basal forebrain",
    "snr"="signal-to-noise ratio",
    "mtl"=" medial temporal lobe",
    "alec"="anterolateral entorhinal cortex",
    "_pmecmtl"="posteromedial entorhinal cortex"
  )
  
  for (pat in names(replacements)) {
    x <- str_replace_all(x, pat, replacements[[pat]])
  }
  
  x <- str_squish(x)
  ifelse(x=="", NA, x)
}


zz$Anatomy = clean_anatomy( zz$Anatomy )
# --------- Write data dictionary (your destination) ---------------------------
# write.csv(zz, "./docs/antspymm_data_dictionary.csv", row.names = FALSE)

# --------- Done message -------------------------------------------------------
message("Wrote antspymm_data_dictionary.csv")
