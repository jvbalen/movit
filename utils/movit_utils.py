import os

import gin
import torch


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


def param_dict(fn_name):
    d = {}
    s = gin.operative_config_str()
    for line in s.split('\n'):
        line = line.strip()
        if line.startswith('#') or len(line) == 0:
            continue
        param, value = line.split(' = ')
        scope, param = param.split('.') 
        if param == fn_name:
            d[param] = value
    return d
