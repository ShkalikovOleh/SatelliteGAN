import os
from glob import glob

import numpy as np
from osgeo import gdal
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose, ConvertImageDtype


class CropsDataset(Dataset):

    def __init__(self, root_dir, exclude_names=None, apply_minmax=True):
        super().__init__()

        self.files = glob(os.path.join(root_dir, '*.tif'))

        if exclude_names is not None:
            for name in exclude_names:
                path = os.path.join(root_dir, name)
                self.files.remove(path)

        self.min_max = apply_minmax
        # assume that max value is 2 if don't apply min max
        param = 0.5 if apply_minmax else 1
        params = [param for _ in range(4)]

        self.norm = Compose([
            ToTensor(),
            ConvertImageDtype(torch.float32),
            Normalize(params, params)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tiff = gdal.Open(self.files[idx])
        h, w = tiff.RasterXSize, tiff.RasterYSize

        image = np.empty((h, w, 4))
        for i in range(1, 5):
            image[..., i - 1] = tiff.GetRasterBand(i).ReadAsArray()

        if self.min_max:
            image = image / np.amax(image, axis=(0, 1))
        image = self.norm(image)

        classes = torch.from_numpy(
            tiff.GetRasterBand(5).ReadAsArray()).to(torch.long)

        mask = F.one_hot(classes - 1,
                         num_classes=19).permute(2, 0, 1).to(torch.float)

        return image, mask
