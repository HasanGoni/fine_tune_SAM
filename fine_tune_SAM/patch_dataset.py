# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_patch_dataset.ipynb.

# %% auto 0
__all__ = ['DPI', 'show_rand_img', 'patch_img_and_mask']

# %% ../nbs/01_patch_dataset.ipynb 4
from fastcore.all import *
import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from patchify import patchify
from scipy import ndimage
import matplotlib as mpl
from typing import List, Tuple, Union, Optional

# %% ../nbs/01_patch_dataset.ipynb 5
DPI=mpl.rcParams['figure.dpi']
mpl.rcParams['image.cmap'] = 'gray'

# %% ../nbs/01_patch_dataset.ipynb 16
def show_rand_img(
        idx:int,  # in case of None, a random image is chosen
        im_path:str, 
        msk_path:str
        ):
    "Show random mask and image from together from path"
    images = Path(im_path).ls()
    masks = Path(msk_path).ls()


    if idx is None:
        idx = np.random.choice(len(images),1)[0]
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    img = cv2.imread(str(images[idx]))
    msk = cv2.imread(str(masks[idx]))
    ax[0].imshow(img,cmap='gray')
    ax[1].imshow(msk,cmap='gray');

# %% ../nbs/01_patch_dataset.ipynb 32
def patch_img_and_mask(
        im_path:Union[Path, str],
        msk_path:Union[Path, str],
        patch_im_path:Union[Path, str],
        patch_msk_path:Union[Path, str],
        PATCH_SIZE:int=256,
        STEP:int=256,
        )->None:

    "Create patch images and masks from original images and masks"
    for i in tqdm(Path(im_path).ls()):
        name = Path(i).name
        msk_name = f'{msk_path}/{name}'
        img=cv2.imread(str(i), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(msk_name, cv2.IMREAD_GRAYSCALE)

        #  patchify img and msks
        patch_img = patchify(img, PATCH_SIZE, STEP)
        patch_img = patch_img.reshape(-1, PATCH_SIZE, PATCH_SIZE)
        # saving patch images
        for patch_,img_ in enumerate(patch_img):
            cv2.imwrite(f'{patch_im_path}/{Path(name).stem}_p_{patch_}.png', img_)

        patch_msk = patchify(msk, PATCH_SIZE, STEP)
        patch_msk= patch_msk.reshape(-1, PATCH_SIZE, PATCH_SIZE)
        for patch_m,m in enumerate(patch_msk):
            cv2.imwrite(f'{patch_msk_path}/{Path(name).stem}_p_{patch_m}.png', m)
   


