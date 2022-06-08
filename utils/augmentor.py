import torch

from utils.tiff import save_to_tiff


class Augmentor:

    def __init__(self, dir_path, model):
        self.model = model
        self.dir = dir_path
        self.device = model.device

        self.N = 0

    def __call__(self, src_idx, dest_idx, mask):
        mask = mask.to(self.device).unsqueeze(dim=0)

        fake = self.model(mask)
        aug = torch.cat([fake, mask], dim=1)[0]

        self.N += 1
        path = f'{self.dir}/{self.N}_{src_idx}_{dest_idx}.tif'
        save_to_tiff(aug, path)
