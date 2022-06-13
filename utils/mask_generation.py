from collections import deque

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def adjust_counts(dataset, current_ppc, dest_ppc, use_mask_func,
                  n_change=2, max_iteration=5):
    current_ppc = current_ppc.clone()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    classes = []
    for k, (cur, dest) in enumerate(zip(current_ppc, dest_ppc)):
        if cur < dest:
            classes.append(k)

    queue = deque(classes)

    i = 0
    loader_iter = iter(loader)
    while len(queue) > 0:
        try:
            _, masks = next(loader_iter)
        except StopIteration:
            i += 1
            if max_iteration == i:
                break
            loader_iter = iter(loader)
            _, masks = next(loader_iter)
        mask = masks[0]

        max_idxs = mask.sum(dim=(1, 2)).argsort()[-n_change:]
        mask_idx = mask.argmax(dim=0)

        for max_idx in max_idxs:
            if queue:
                new_idx = queue.popleft()
                if new_idx in max_idxs:
                    queue.append(new_idx)
                    continue
            else:
                break

            cond = (mask_idx == max_idx)
            mask_idx = torch.where(cond, torch.tensor(new_idx), mask_idx)

            if current_ppc[new_idx] + cond.sum() < dest_ppc[new_idx]:
                queue.append(new_idx)

        mask = F.one_hot(mask_idx, num_classes=19).permute(2, 0, 1).float()
        use_mask_func(mask)
        current_ppc += mask.sum(dim=(1, 2)).long()

    return current_ppc
