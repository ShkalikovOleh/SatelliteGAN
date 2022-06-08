import torch
from osgeo import gdal


def save_to_tiff(data, path, norm=True):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    ch, h, w = data.shape

    if norm:
        data = ((data + 1) / 2)

    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(path, h, w, ch, gdal.GDT_Float32)

    for i in range(ch):
        output.GetRasterBand(i+1).WriteArray(data[i])

    output.FlushCache()
