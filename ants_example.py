# Skull strip with ANTs


# rewrite of https://github.com/Angeluz-07/MRI-preprocessing-techniques/blob/main/notebooks/09_brain_extraction_with_template.ipynb
import os
import util
import ants
import numpy as np
import neurovolume as nv

util.download_test_dataset()
template = ants.image_read("./data/mni_icbm152_t1_tal_nlin_sym_09a.nii", reorient="IAL")
template_mask = ants.image_read(
    "./data/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii", reorient="IAL"
)

anat = ants.image_read("./data/sub-01_T1w.nii")  # might not work!
np.unique(template_mask.numpy())  # ?

print("registering template to anat")
transformation = ants.registration(
    fixed=anat, moving=template, type_of_transform="SyN", verbose=True
)
print("registering matte to template")
brain_mask = ants.apply_transforms(
    fixed=transformation["warpedmovout"],
    moving=template_mask,
    transformlist=transformation["fwdtransforms"],
    interpolator="nearestNeighbor",
    verbose=True,
)

print("dialating brain mask...")
brain_mask_dilated = ants.morphology(
    brain_mask, radius=4, operation="dilate", mtype="binary"
)
print("masking anat...")
masked = ants.mask_image(anat, brain_mask_dilated)

print("saving vdb...")
print("normalizing array...")
norm = util.normalize_array(masked.numpy()).astype(order="C", dtype=np.float64)

norm = np.transpose(norm, (1, 2, 0))
norm = np.ascontiguousarray(norm)


print(f"sanity check: this should be an ndarray: {type(norm)} dtype: {norm.dtype}")
print(f"Does ./data/ exist? {os.path.exists('./data/')}")
print(f"Array shape: {norm.shape}")
print(f"Array contains NaN? {np.isnan(norm).any()}")
print(f"Array contains Inf? {np.isinf(norm).any()}")

nv.ndarray_to_VDB(norm, "./data/masked.vdb")  # BUG:
#  something broken here

print("done")
