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
zz = data.frame(Label = colnames(dd), stringsAsFactors = FALSE)

# Identify QC rows exactly as you did
qcrows = min(grep("RandBasis", zz$Label)) : grep("resnetGrade", zz$Label)

# Base columns (follow your assignments)
zz$Modality = "Other"
zz[grep("T1Hier",  zz$Label), "Modality"] = "T1 hierarchical processing"
zz[grep("T1w",     zz$Label), "Modality"] = "T1 DiReCT thickness processing"
zz[grep("DTI",     zz$Label), "Modality"] = "DTI"
zz[grep("NM2DMT",  zz$Label), "Modality"] = "Neuromelanin"
zz[grep("rsfMRI",  zz$Label), "Modality"] = "restingStatefMRI"
zz[grep("lair|flair", zz$Label, ignore.case = TRUE), "Modality"] = "Flair"

zz$side = NA
zz[grep("left",  zz$Label, ignore.case = TRUE), "side"]  = "left"
zz[grep("right", zz$Label, ignore.case = TRUE), "side"]  = "right"

zz$Atlas = "ANTs"
zz[grep("dkt",     zz$Label, ignore.case = TRUE), "Atlas"] = "desikan-killiany-tourville"
zz[grep("cnxcou",  zz$Label, ignore.case = TRUE), "Atlas"] = "desikan-killiany-tourville"
zz[grep("jhu",     zz$Label, ignore.case = TRUE), "Atlas"] = "johns hopkins white matter"
zz[grep("cit",     zz$Label, ignore.case = TRUE), "Atlas"] = "CIT168"
zz[grep("nbm",     zz$Label, ignore.case = TRUE), "Atlas"] = "BF"
zz[grep("ch13",    zz$Label, ignore.case = TRUE), "Atlas"] = "BF"
zz[grep("mtl",     zz$Label, ignore.case = TRUE), "Atlas"] = "MTL"
zz[grep("rsfMRI",  zz$Label, ignore.case = TRUE), "Atlas"] = "power peterson fMRI meta-analyses"
zz[qcrows, "Atlas"] = "quality control metrics"

zz$Measurement = NA
zz[grep("FD",      zz$Label, ignore.case = TRUE), "Measurement"] = "motion statistic on framewise displacement"
zz[grep("thk",     zz$Label, ignore.case = TRUE), "Measurement"] = "geometry/thickness"
zz[grep("area",    zz$Label, ignore.case = TRUE), "Measurement"] = "geometry/area"
zz[grep("vol",     zz$Label, ignore.case = TRUE), "Measurement"] = "geometry/volume"
zz[grep("mean_md", zz$Label, ignore.case = TRUE), "Measurement"] = "mean diffusion"
zz[grep("mean_fa", zz$Label, ignore.case = TRUE), "Measurement"] = "fractional anisotropy"
zz[grep("cnx",     zz$Label, ignore.case = TRUE), "Measurement"] = "tractography-based connectivity"

# Start Anatomy as the label (your pattern), then strip measurement tokens
zz$Anatomy = zz$Label
zz$Anatomy = gsub("_thk_",       "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("_area_",      "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("_volume_",    "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("DTI_cnxcount","", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("DTI_mean_md", "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("DTI_mean_fa", "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("T1Hier_",     "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("T1Hier",      "", zz$Anatomy, ignore.case = TRUE)

# --------- DKT mapping (yours, with case-insensitive matching) ----------------
dktlabs  = dktcsv$Description
dktlabs  = gsub("^right\\s+|^left\\s+", "", dktlabs, ignore.case = TRUE)
dktlabs2 = gsub(" ", "_", dktlabs)

for (k in seq_along(dktlabs)) {
  hits = grep(dktlabs[k],  zz$Label, ignore.case = TRUE)
  if (length(hits)) {
    zz[hits, "Atlas"]   = "desikan-killiany-tourville"
    zz[hits, "Anatomy"] = dktlabs[k]
  }
  hits2 = grep(dktlabs2[k], zz$Label, ignore.case = TRUE)
  if (length(hits2)) {
    zz[hits2, "Atlas"]   = "desikan-killiany-tourville"
    zz[hits2, "Anatomy"] = dktlabs[k]
  }
}

# --------- CIT mapping (yours, but case-insensitive against label) ------------
citlabs = tolower(cit$Description)
for (k in seq_along(citlabs)) {
  gg = grep(citlabs[k], tolower(zz$Label), ignore.case = FALSE)
  if (length(gg)) {
    zz[gg, "Atlas"]   = "CIT168"
    zz[gg, "Anatomy"] = cit$Anatomy[k]
  }
}

# --------- Post-clean gsubs (yours) -------------------------------------------
zz$Anatomy = gsub("DTIfa", "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("DTImd", "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("dktregions", "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("dktcortex", " cortex only ", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("_right_", "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("_left_",  "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("right",   "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("left",    "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("jhu_icbm_labels_1mm", "", zz$Anatomy, ignore.case = TRUE)

# Unique id rows
if (length(grep("u_hier_id", zz$Label))) {
  zz[grep("u_hier_id", zz$Label), setdiff(colnames(zz), "Label")] = "unique id"
}

# Interpret DTI connectivity rows (yours)
cnxrows = grep("DTI_cnxcount", zz$Label, ignore.case = TRUE)
for (k in cnxrows) zz$Anatomy[k] = interpretcnx(zz[k, "Label"])

# --------- rsfMRI side + measurement granularity (your logic, stronger) -------
if (!exists("multigrep")) {
  # local fallback already defined above
}

# side by explicit R/L token next to rsfMRI (robust to underscores)
r_right = multigrep(c("rsfMRI", "(^|_)R(_|$)"), zz$Label, intersect = TRUE)
r_left  = multigrep(c("rsfMRI", "(^|_)L(_|$)"), zz$Label, intersect = TRUE)
if (length(r_right)) zz[r_right, "side"] = "right"
if (length(r_left))  zz[r_left,  "side"] = "left"

# rsfMRI measurement types
zz$Measurement[multigrep(c("rsfMRI","_2_"),    zz$Label, intersect = TRUE)] = "network correlation"
zz$Measurement[multigrep(c("rsfMRI","_alff"),  zz$Label, intersect = TRUE)] = "amplitude of low frequency fluctuations ALFF"
zz$Measurement[multigrep(c("rsfMRI","_falff"), zz$Label, intersect = TRUE)] = "fractional amplitude of low frequency fluctuations fALFF"

# Strip rsfMRI scaffolding from Anatomy
zz$Anatomy = gsub("^rsfMRI_",    "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("falffPoint",  "", zz$Anatomy, ignore.case = TRUE)
zz$Anatomy = gsub("alffPoint",   "", zz$Anatomy, ignore.case = TRUE)

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
zz$FcProc   = NA
zz$FcProc[is_rsfmri] = sapply(zz$Label[is_rsfmri], extract_fcnxpro)

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
  zz$Anatomy <- gsub(pat, acro_map[[pat]], zz$Anatomy, ignore.case = TRUE, perl = TRUE)
}

# Whitespace tidy
zz$Anatomy <- gsub("\\s+", " ", zz$Anatomy)
zz$Anatomy <- trimws(zz$Anatomy)

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

# --------- NeedsIntervention flag for curation --------------------------------
NeedsIntervention <- rep(FALSE, nrow(zz))

# Flag if rsfMRI row failed to decode pair (but looked like a pair)
looks_like_pair <- grepl("_2_", zz$Label) & grepl("rsfMRI", zz$Label, ignore.case = TRUE)
NeedsIntervention[looks_like_pair & is.na(decoded_pairs)] <- TRUE

# Flag if CIT label but no CIT anatomy resolved
is_cit <- grepl("cit", zz$Label, ignore.case = TRUE)
NeedsIntervention[is_cit & is.na(match(zz$Anatomy, unique(na.omit(cit$Anatomy))))] <- TRUE

# Flag if Anatomy still contains obvious scaffolding tokens or leftover codes
leftover_tokens <- c("fcnxpro", "alffPoint", "falffPoint", "dktregions", "jhu_icbm_labels_1mm")
for (tok in leftover_tokens) {
  NeedsIntervention[grepl(tok, zz$Anatomy, ignore.case = TRUE)] <- TRUE
}

# Flag if Anatomy ended up empty-like
NeedsIntervention[nchar(trimws(zz$Anatomy)) == 0] <- TRUE

zz$NeedsIntervention = NeedsIntervention

zz$Anatomy=gsub("fcnxpro122","",zz$Anatomy,ignore.case=TRUE)
zz$Anatomy=gsub("fcnxpro134","",zz$Anatomy,ignore.case=TRUE)
zz$Anatomy=gsub("fcnxpro129","",zz$Anatomy,ignore.case=TRUE)
zz$Anatomy=gsub("_"," ",zz$Anatomy,ignore.case=TRUE)
zz$Anatomy=gsub("bf"," basal forebrain",zz$Anatomy,ignore.case=TRUE)
zz$Anatomy=gsub("_nbm_"," basal forebrain ",zz$Anatomy,ignore.case=TRUE)

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
write.csv(zz, "./docs/antspymm_data_dictionary.csv", row.names = FALSE)

# --------- Done message -------------------------------------------------------
message("Wrote ~/code/ANTsPyMM/antspymm_data_dictionary.csv with ",
        sum(zz$NeedsIntervention), " rows flagged for review.")