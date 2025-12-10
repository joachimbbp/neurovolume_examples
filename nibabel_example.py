import nibabel as nib
import neurovolume as nv
import numpy as np
import util

file = "./data/sub-01_T1w.nii.gz"
util.download_test_dataset()


print("loading data...")
img = nib.load(file)
data = np.array(img.get_fdata(), order="C", dtype=np.float64)
print("preparing data...")
prepped_data = nv.prep_ndarray(data)

output = "./data/from_nib.vdb"
print("creating vdb...")
nv.ndarray_to_VDB(prepped_data, output, img.affine)
print("data saved as vdb")
print("done")
