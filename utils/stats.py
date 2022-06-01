import torch


def counts_per_classes(loader):
    '''
    Return number of pixels and number of images per class
    '''

    _, masks = next(iter(loader))
    n_classes = masks.shape[1]

    pixels_counts = torch.zeros(n_classes, dtype=torch.long)
    images_counts = torch.zeros(n_classes, dtype=torch.long)

    for _, masks in loader:
        bs, ch, h, w = masks.shape
        # n images per item and class
        ipbc = masks.view(bs, ch, h * w).any(dim=2)
        images_counts += ipbc.sum(dim=0).to(torch.long)
        pixels_counts += masks.sum(dim=(0, 2, 3)).to(torch.long)

    return pixels_counts, images_counts


def mean_std_per_classes(loader):
    images, masks = next(iter(loader))
    n_classes = masks.shape[1]
    n_channels = images.shape[1]

    n = torch.zeros(n_classes)  # pixel counts per class
    s = torch.zeros(n_classes, n_channels)  # total sum for each channel
    sq_s = torch.zeros(n_classes, n_channels)  # total squared sum

    for images, masks in loader:
        n += masks.sum(dim=(0, 2, 3))
        for j in range(n_classes):
            temp = images * masks[:, j].unsqueeze(1)
            s[j] += temp.sum(dim=(0, 2, 3))
            sq_s[j] += torch.pow(temp, 2).sum(dim=(0, 2, 3))

    n = n.unsqueeze(1)
    mean = s / n
    std = torch.sqrt((sq_s - n * mean**2) / (n - 1))

    return mean, std


def hist_per_channels(loader, bins=10, min=-1, max=1):
    images, _ = next(iter(loader))
    n_channels = images.shape[1]

    result = torch.zeros(n_channels, bins, dtype=torch.long)

    for images, _ in loader:
        for i in range(n_channels):
            data = images[:, i].flatten()
            result[i] += torch.histc(data, bins=bins,
                                     min=min, max=max).to(torch.long)

    return result
