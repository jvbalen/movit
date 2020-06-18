import os
import datetime

import torch

from torch.utils.data import DataLoader

from dataset.move_dataset_fixed_size import MOVEDatasetFixed
from dataset.move_dataset_full_size import MOVEDatasetFull
from utils.move_utils import triplet_mining_collate


def load_data(path, n_bins, patch_len=1800, data_aug=True, fixed_size=True, num_of_labels=16, ytc=False):
    """
    initializing the MOVE dataset objects and data loaders
    we use validation set to track two things, (1) triplet loss, (2) mean average precision
    to check mean average precision on the full songs,
    we need to define another dataset object and data loader for it

    :param train_path: path of the training data
    :param val_path: path of the validation data
    :param n_bins: number of bins (23 for MODEModel, 12 for MOVEModelNT)
    :param patch_len: number of frames for each song to be used in training
    :param data_aug: whether to use data augmentation
    :param num_of_labels: number of labels per mini-batch
    :param ytc: whether to exclude the songs overlapping with ytc for training
    """
    data, labels = import_dataset_from_pt(path, n_bins=n_bins)

    if fixed_size:
        dataset = MOVEDatasetFixed(data, labels, h=n_bins, w=patch_len,
                                    data_aug=data_aug, ytc=ytc)
        loader = DataLoader(dataset, batch_size=num_of_labels, shuffle=True,
                            collate_fn=triplet_mining_collate, drop_last=True)
    else:
        dataset = MOVEDatasetFull(data, labels)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

    return loader, len(data)


def import_dataset_from_pt(path, n_bins=23):
    """
    Loading a dataset stored in .pt format
    :param path: name of the .pt file to load
    :param n_bins: number of bins (23 for MODEModel, 12 for MOVEModelNT)
    :return: lists that contain data and labels (elements are in the same order)
    """
    if os.path.isdir(path):
        filenames = [f for f in os.listid(path) if f.endswith('.pt')] 
    else:
        filenames = [path]

    data, labels = list(), list()
    for filename in filenames:
        dataset_dict = torch.load(filename)
        data.extend(dataset_dict['data'])
        labels.extend(dataset_dict['labels'])

    if n_bins < 23:  # depending on n_bins of model type, reshape the pcp features
        data = [data[i][:, :n_bins] for i in range(len(data))]

    return data, labels


def make_log_dir(base_dir):

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_dir, timestamp)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    return log_dir