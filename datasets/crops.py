from glob import glob

import numpy as np
from osgeo import gdal
import torch
import torch.nn. functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose


class CropsDataset(Dataset):

    def __init__(self, root_dir):
        super().__init__()

        self.files = glob(f'{root_dir}/*.tif')
        self.norm = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5, 0.5),
                      (0.5, 0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tiff = gdal.Open(self.files[idx])
        h, w = tiff.RasterXSize, tiff.RasterYSize

        image = np.empty((h, w, 4))
        for i in range(1, 5):
            image[..., i-1] = tiff.GetRasterBand(i).ReadAsArray()
        image = self.norm(image)

        classes = torch.from_numpy(
            tiff.GetRasterBand(5).ReadAsArray()).to(torch.long)
        mask = F.one_hot(classes)

        return image, mask
