import os.path as osp
import cv2

import numpy as np
from itertools import chain
import os
import os.path as osp
import shutil
from itertools import chain
from xml.dom import minidom
try:
    import torchvision.models as models
    import torchvision.transforms as V
    from PIL import Image
except ImportError:
    models = None
    T = None
    Image = None

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar)

from pascalVOCSingle import PascalVOCSingleObject
import torch_geometric.transforms as T

from dgmc.utils import ValidPairDataset

pre_filter = lambda data: data.pos.size(0) > 0  # noqa
transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Distance() if False else T.Cartesian(),
])

train_datasets = []
test_datasets = []
path = osp.join('..', 'data', 'PascalVOC')
for category in PascalVOCSingleObject.categories:
        if category != 'chair':
            dataset_single = PascalVOCSingleObject(path, category, train=True, transform=transform,
                                pre_filter=pre_filter)
            train_datasets += [ValidPairDataset(dataset_single, dataset_single, sample=True)]
            dataset = PascalVOCSingleObject(path, category, train=False, transform=transform,
                        pre_filter=pre_filter)
            test_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, 16, shuffle=True,
                          follow_batch=['x_s', 'x_t'])
