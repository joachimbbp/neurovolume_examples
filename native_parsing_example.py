from urllib.request import urlretrieve
import neurovolume as nv
import gzip
import shutil

anat_url = "https://s3.amazonaws.com/openneuro.org/ds003548/sub-01/anat/sub-01_T1w.nii.gz?versionId=5ZTXVLawdWoVNWe5XVuV6DfF2BnmxzQz"
anat_gz = "./data/sub-01_T1w.nii.gz"
anat = "./data/sub-01_t1w.nii"
print("Downloading test data...")
urlretrieve(anat_url, anat_gz)
print("Test data downloaded")
print("Unzipping...")
with gzip.open(anat_gz, "rb") as f_in:
    with open(anat, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


output = "./data/"
print("writing vdb...")
vdb_path = nv.nifti1_to_VDB(anat, output, True)
print("vdb written to ", vdb_path)
