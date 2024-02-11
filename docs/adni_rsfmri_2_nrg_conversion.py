
import glob
import os
import shutil
import subprocess

# Set the directory paths and pattern for the files
fns = glob.glob("ADNI/*/*/*/*")

for k in range(len(fns)):
    pid = 'ADNI'
    modality = 'rsfMRI'
    fn = os.path.splitext(fns[k])[0]
    uid = os.path.basename(fn)
    fn = fn.replace("ADNI/", "", 1)
    temp0 = fn.split('/')
    sid = temp0[0]
    dt = temp0[2].replace("-", "")[:10]
    dt = dt.split("_")[0]
    dcm2cmd = f"dcm2niix -z y {fns[k]}"
    mynii = glob.glob(f"{fns[k]}/*nii.gz")
    rdir = '/mnt/cluster/data/ADNI/nrg/ADNI'
    odir = '/'.join([rdir, sid, dt, modality, uid])            
    myfn = '-'.join([pid, sid, dt, modality, uid])
    temp = os.path.join(odir, f"{myfn}.nii.gz")
    if not os.path.exists(temp):
        print("Get " + temp )
        subprocess.run(dcm2cmd, shell=True)
        mynii = glob.glob(f"{fns[k]}/*nii.gz")
        if len( mynii ) > 0:
            checkfns = [
                mynii[0],
                mynii[0].replace("nii.gz", "json")
            ]
            outfns = [
                os.path.join(odir, f"{myfn}.nii.gz"),
                os.path.join(odir, f"{myfn}.json")
            ]
            os.makedirs(odir, exist_ok=True)
            if all(os.path.exists(checkfn) for checkfn in checkfns):
                    for zz in range(len(outfns)):
                        if zz == 0:
                            print("GOTIT")
                            print(outfns[zz])
                        shutil.move(checkfns[zz], outfns[zz])


