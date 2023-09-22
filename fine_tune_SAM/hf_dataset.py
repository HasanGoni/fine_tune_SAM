# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_upload_data_hf.ipynb.

# %% auto 0
__all__ = ['show_hf_dataset']

# %% ../nbs/03_upload_data_hf.ipynb 3
from typing import List, Dict, Union, Optional
from datasets import load_dataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'gray' 

# %% ../nbs/03_upload_data_hf.ipynb 5
def show_hf_dataset(
        dataset:Dataset,
        idx:Union[int, None]=None,
        split:str='train'
        ):
    "Show hugging face random index"

    if idx is None:
        idx = np.random.randint(0, len(dataset[split]))

    print(f' dataset index will be visualized: {idx}')
    im_ = dataset[split]['image'][idx]
    msk_ = dataset[split]['label'][idx]
    fig, ax = plt.subplots(
        1, 2, figsize=(10, 5)
    )
    ax[0].imshow(im_)
    ax[0].set_title('image')
    ax[0].axis('off')
    ax[1].imshow(msk_)
    ax[1].set_title('mask')
    ax[1].axis('off')
    

