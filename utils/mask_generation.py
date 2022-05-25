from collections import deque

from torch.utils.data import DataLoader


def adjust_counts(dataset, current_ppc, dest_ppc, use_mask_func):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    classes = []
    for k, (cur, dest) in enumerate(zip(current_ppc, dest_ppc)):
        if cur < dest:
            classes.append(k)

    queue = deque(classes)

    loader_iter = iter(loader)
    while len(queue) > 0:
        try:
            _, masks = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            _, masks = next(loader_iter)
        mask = masks[0]

        max_idx = mask.sum(dim=(1, 2)).argmax()
        cond = mask[max_idx] == 1

        mask[max_idx, cond] = 0

        idx = queue.popleft()
        mask[idx, cond] = 1

        use_mask_func(max_idx, idx, mask)

        current_ppc[idx] += cond.sum()
        if current_ppc[idx] < dest_ppc[idx]:
            queue.append(idx)
