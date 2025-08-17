import matplotlib.pyplot as plt
import copy
from glob import glob
import itertools
import math
import os
from pathlib import Path
import shutil
from typing import Callable, Literal, Optional, Type
import warnings

from tqdm import tqdm
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd

# import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Subset
# from torch.nn import functional as F
from torchvision import datasets, transforms, models
# from torchvision.utils import save_image, make_grid

def create_STL10_dateloader(
        split: str,
        dir_to_save: str = "DIR_DATA",
        transform: Tensor = eval_transforms,
        label_to_use: list[str] = None,
        all_label: list[str] = None,
) -> DataLoader:
    dataset = datasets.STL10(
        root = dir_to_save,
        split = split,
        download = True,
        transform = transform,
    )

    if set(label_to_use) != set(dataset.classes):
        indices_to_use = {
            dataset.classes.index(cat) for cat in  label_to_use
        }
        indices = [
            i for i, label in enumerate(dataset.labels)
            if label in indices_to_use
        ]
        dataset = Subset(dataset, indices)
        dataset.clasess = all_label

    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True if split == "train" else False,
        num_workers = os.cpu_count(),
    )

def show_images_od_first_batch(dataloader: DataLoader, batch_size: int):
    torch.manual_seed(0)
    X, y = next(iter(dataloader))
    nrow = 4
    plt.figure(figsize=(
        2 * nrow,
        2 * math.ceil(batch_size / nrow)
    ))
    plt.imshow(
        make_grid(destanderdize(X), nrow=nrow, padding=8, pad_value=1)
        .permute(1, 2, 0)
    )
