from image_def_03 import (
    create_STL10_dateloader,
    show_images_od_first_batch,
)

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
import seaborn as sns
import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid

eval_transforms = transforms.Compose([
    transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

BATCH_SIZE = 64
IMAGE_SIZE = 256  # 一辺のピクセル数
MAX_EPOCHS = 30  # 学習時の最大エポック数

# ImageNet統計に基づく正規化パラメータ
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ROOT_DIR_PATH = Path('/app/04/')
DIR_DATA = ROOT_DIR_PATH / "data"
DIR_DATA_UNKNOWN = DIR_DATA / "unknown"
DIR_MODELS = ROOT_DIR_PATH / "models"

os.makedirs(DIR_DATA, exist_ok=True)
os.makedirs(DIR_DATA_UNKNOWN, exist_ok=True)
os.makedirs(DIR_MODELS, exist_ok=True)

ALL_LABELS = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]
LABELS_TO_USE = ["bird", "cat", "deer", "dog", "horse", "monkey"]

NUM_CLASSES = len(ALL_LABELS)
LABELS_MAP = {
    i: LABELS_TO_USE.index(label)
    for i, label in enumerate(ALL_LABELS) if label in LABELS_TO_USE
}

dataloader_for_display = create_STL10_dateloader(
    split = "train",
    dir_to_save = DIR_DATA,
    label_to_use = LABELS_TO_USE,
    all_label = ALL_LABELS,
)
show_images_od_first_batch(
    dataloader_for_display,
    batch_size = BATCH_SIZE,
)

