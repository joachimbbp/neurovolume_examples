import nibabel as nib
import neurovolume as nv
import numpy as np
import util

file = "./data/sub-01_T1w.nii.gz"
util.download_test_dataset()


img = nib.load(file)
data = np.array(img.get_fdata(), order="C", dtype=np.float64)
print("data loaded")
norm = util.normalize_array(data).astype(np.float64)

norm = np.transpose(norm, (1, 2, 0))
norm = np.ascontiguousarray(norm)
print("data normalized")
print(f"sanity check: this should be an ndarray: {type(norm)} dtype: {norm.dtype}")

output = "./data/from_nib.vdb"
print("creating vdb...")
nv.ndarray_to_VDB(norm, output, img.affine)
print("data saved as vdb")
print("done")
