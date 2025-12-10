# Skull strip with ANTs
# saves the skull and the brain separately
# based on https://github.com/Angeluz-07/MRI-preprocessing-techniques/blob/main/notebooks/09_brain_extraction_with_template.ipynb

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

print("registering template to anat...")
transformation = ants.registration(
    fixed=anat, moving=template, type_of_transform="SyN", verbose=True
)
print("registering matte to template...")
brain_mask = ants.apply_transforms(
    fixed=transformation["warpedmovout"],
    moving=template_mask,
    transformlist=transformation["fwdtransforms"],
    interpolator="nearestNeighbor",
    verbose=True,
)

print("dilating brain mask...")
brain_mask_dilated = ants.morphology(
    brain_mask, radius=4, operation="dilate", mtype="binary"
)
print("masking skull...")
brain_isolated_raw = ants.mask_image(anat, brain_mask_dilated)
print("preparing brain...")
brain_isolated = nv.prep_ndarray(brain_isolated_raw.numpy())
print("saving brain vdb...")
nv.ndarray_to_VDB(brain_isolated, "./data/brain.vdb")

inverted_mask = 1 - brain_mask_dilated # LLM
print("masking brain...")
print(f"type of numpy")
skull_isolated_raw = ants.mask_image(anat, inverted_mask)
print("preparing skull...")
skull_isolated = nv.prep_ndarray(skull_isolated_raw.numpy())
print("saving anat vdb...")
nv.ndarray_to_VDB(skull_isolated, "./data/skull.vdb")

print("done")
