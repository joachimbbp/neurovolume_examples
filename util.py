import os
from urllib.request import urlretrieve
import numpy as np


def download_if_absent(link, dir="./data"):
    file = f"{dir}/{link.split('/')[-1]}"
    if os.path.exists(file):
        print(f"{file} already exissts")
    else:
        print(f"Downloading {file}")
        urlretrieve(link, file)


# NOTE: needs some more coverage and unzipping

# NOTE: maybe this should go in Neurovolume library?????
# both the download_if_absent as well as an accessible csv or whatever
# of the test file links would be useful!


def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


# this too!


def download_test_dataset():
    links = [
        "https://s3.amazonaws.com/openneuro.org/ds003548/sub-01/anat/sub-01_T1w.nii.gz",
        "https://github.com/Angeluz-07/MRI-preprocessing-techniques/raw/refs/heads/main/assets/templates/mni_icbm152_t1_tal_nlin_sym_09a.nii",
        "https://github.com/Angeluz-07/MRI-preprocessing-techniques/raw/refs/heads/main/assets/templates/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii",
    ]
    for link in links:
        download_if_absent(link)
    # maybe this could return a filepath to allow for seamless loading? Might get a little fancy
