import random

import numpy as np
import pandas as pd
import torch
from scipy import interpolate


def import_dataset_from_pt(filename, chunks=1, model_type=0):
    """
    Loading a dataset stored in .pt format
    :param filename: name of the .pt file to load
    :param chunks: number of chunks to load
    :param model_type: type of the model to use (related to pcp feature size)
    :return: lists that contain data and labels (elements are in the same order)
    """

    if chunks > 1:
        for i in range(1, chunks+1):
            dataset_dict = torch.load('{}_{}.pt'.format(filename, i))
            if i == 1:
                data = dataset_dict['data']
                labels = dataset_dict['labels']
            else:
                data.extend(dataset_dict['data'])
                labels.extend(dataset_dict['labels'])
    else:
        dataset_dict = torch.load('{}'.format(filename))
        data = dataset_dict['data']
        labels = dataset_dict['labels']

    if model_type == 1:  # depending on the model type, reshape the pcp features
        data = [data[i][:, :12] for i in range(len(data))]

    return data, labels


def cs_augment(pcp, p_pitch=1, p_stretch=0.3, p_warp=0.3):
    """
    Applying data augmentation to a given pcp patch
    :param pcp: pcp patch to augment (dimensions should be 1 x H x W)
    :param p_pitch: probability of applying pitch transposition
    :param p_stretch: probability of applying time stretch (with linear interpolation)
    :param p_warp: probability of applying time warping (silence, duplication, removal)
    :return: augmented pcp patch
    """
    pcp = pcp.cpu().detach().numpy()  # converting the pcp patch to a numpy matrix

    # pitch transposition
    if torch.rand(1) <= p_pitch:
        shift_amount = torch.randint(low=0, high=12, size=(1,))
        pcp_aug = np.roll(pcp, shift_amount, axis=1)  # applying pitch transposition
    else:
        pcp_aug = pcp

    _, h, w = pcp_aug.shape
    times = np.arange(0, w)  # the original time stamps

    # interpolation function for time stretching and warping
    func = interpolate.interp1d(times, pcp_aug, kind='nearest', fill_value='extrapolate')

    # time stretch (JVB: replaced some torch with numpy here or this would break)
    if torch.rand(1) < p_stretch:
        p = random.random()  # random number to determine the factor of time stretching
        if p <= 0.5:
            times_aug = np.linspace(0, w - 1, int(w * np.clip((1 - p), 0.7, 1)))
        else:
            times_aug = np.linspace(0, w - 1, int(w * np.clip(2 * p, 1, 1.5)))
        pcp_aug = func(times_aug)  # applying time stretching
    else:
        times_aug = times
        pcp_aug = func(times_aug)

    # time warping
    if torch.rand(1) < p_warp:
        p = torch.rand(1)  # random number to determine which operation to apply for time warping

        if p < 0.3:  # silence
            # each frame has a probability of 0.1 to be silenced
            silence_idxs = np.random.choice([False, True], size=times_aug.size, p=[.9, .1])
            pcp_aug[:, :, silence_idxs] = np.zeros((h, 1))

        elif p < 0.7:  # duplicate
            # each frame has a probability of 0.15 to be duplicated
            duplicate_idxs = np.random.choice([False, True], size=times_aug.size, p=[.85, .15])
            times_aug = np.sort(np.concatenate((times_aug, times_aug[duplicate_idxs])))
            pcp_aug = func(times_aug)

        else:  # remove
            # each frame has a probability of 0.1 to be removed
            remaining_idxs = np.random.choice([False, True], size=times_aug.size, p=[.1, .9])
            times_aug = times_aug[remaining_idxs]
            pcp_aug = func(times_aug)

    return torch.from_numpy(pcp_aug)  # casting the augmented pcp patch as a torch tensor


def triplet_mining_collate(batch):
    """
    Custom collate function for triplet mining
    :param batch: elements of the mini-batch (pcp features and labels)
    :return: collated elements
    """
    items = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    return torch.cat(items, 0), labels


def average_precision(label_path, ypred, k=None, eps=1e-10, reduce_mean=True, q_file=None, c_file=None):
    """
    Calculating performance metrics
    :param label_path: path to ground truth labels
    :param ypred: square distance matrix
    :param k: k value for map@k
    :param eps: epsilon value for numerical stability
    :param reduce_mean: whether to take mean of the average precision values of each query
    :param q_file: path to txt file listing the rows of ypred and ytrue denoting queries, as one-based indices
    :param c_file: path to txt file listing the cols of ypred and ytrue denoting candidates, as one-based indices
    :return: mean average precision value
    """
    ytrue = torch.load(label_path).float()
    if q_file:
        q = np.loadtxt(q_file) - 1  # one-based -> zero-based
        c = np.loadtxt(c_file) - 1
        ytrue, ypred = ytrue[q][:, c], ypred[q][:, c]

    if k is None:
        k = ypred.size(1)
    _, spred = torch.topk(ypred, k, dim=1)
    found = torch.gather(ytrue, 1, spred)

    temp = torch.arange(k).float() * 1e-6
    _, sel = torch.topk(found - temp, 1, dim=1)
    mrr = torch.mean(1/(sel+1).float())
    mr = torch.mean((sel+1).float())
    top1 = torch.sum(found[:, 0])
    top10 = torch.sum(found[:, :10])

    pos = torch.arange(1, spred.size(1)+1).unsqueeze(0).to(ypred.device)
    prec = torch.cumsum(found, 1)/pos.float()
    mask = (found > 0).float()
    ap = torch.sum(prec*mask, 1)/(torch.sum(ytrue, 1)+eps)
    ap = ap[torch.sum(ytrue, 1) > 0]
    print('mAP: {:.3f}'.format(ap.mean().item()))
    print('MRR: {:.3f}'.format(mrr.item()))
    print('MR: {:.3f}'.format(mr.item()))
    print('Top1: {}'.format(top1.item()))
    print('Top10: {}'.format(top10.item()))
    if reduce_mean:
        return ap.mean()
    return ap


def pairwise_distance_matrix(x, y=None, eps=1e-12):
    """
    Calculating squared euclidean distances between the elements of two tensors
    :param x: first tensor
    :param y: second tensor (optional)
    :param eps: epsilon value for avoiding div by zero
    :return: pairwise distance matrix
    """
    x_norm = x.pow(2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = y.pow(2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2 * torch.mm(x, y.t().contiguous())
    return torch.clamp(dist, eps, np.inf)
