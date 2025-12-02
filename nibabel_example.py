import nibabel as nib
import neurovolume as nv
import numpy as np
from urllib.request import urlretrieve

url = "https://s3.amazonaws.com/openneuro.org/ds003548/sub-01/anat/sub-01_T1w.nii.gz?versionId=5ZTXVLawdWoVNWe5XVuV6DfF2BnmxzQz"
file = "./data/sub-01_T1w.nii.gz"

print("downloading test data...")
f_res = urlretrieve(url, file)

print("static testfile downloaded to ", file)


def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


img = nib.load(file)
data = np.array(img.get_fdata(), order="C", dtype=np.float64)
print("data loaded")
norm = normalize_array(data).astype(np.float64)

norm = np.transpose(norm, (1, 2, 0))
norm = np.ascontiguousarray(norm)
print("data normalized")

output = "./data/from_nib.vdb"
print("creating vdb...")
nv.ndarray_to_VDB(norm, output, img.affine)
print("data saved as vdb")
print("done")
