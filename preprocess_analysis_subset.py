import os
import sys
import json
import random

import deepdish as dd
import numpy as np
import torch


def create_analysis_pt(base_dir, json_path=None, h5_path=None, out_path=None, transposed_pcp=False):
    """
    Function for preprocessing the cremaPCP features of the Da-TACOS analysis subset.
    Preprocessed files are stored in output_dir directory as a single .pt file
    :param output_dir: the directory to store the .pt file
    :return: labels of the processed cremaPCP features
    """
    # reading the metadata file for the Da-TACOS analysis subset
    if json_path is None:
        json_path = 'da-tacos_metadata/da-tacos_coveranalysis_subset_metadata.json'
    if h5_path is None:
        h5_path = 'da-tacos_coveranalysis_subset_crema'
    if out_path is None:
        out_path = 'coveranalysis_crema.pt'
    
    print(f'Creating file {out_path}...')
    with open(os.path.join(base_dir, json_path)) as f:
        metadata = json.load(f)

    data = []
    labels = []
    # iterating through the metadata file to create .pt file
    for key1 in metadata.keys():  # key1 specifies the work id
        for key2 in metadata[key1].keys():  # key2 specifies the performance id
            # loading the file
            temp_path = os.path.join(base_dir, h5_path, '{}_crema/{}_crema.h5'.format(key1, key2))
            # reading cremaPCP features
            temp_crema = dd.io.load(temp_path)['crema']
            if transposed_pcp:
                temp_crema = temp_crema.T

            # downsampling the feature matrix and casting it as a torch.Tensor
            idxs = np.arange(0, temp_crema.shape[0], 8)
            temp_tensor = torch.from_numpy(temp_crema[idxs].T)

            # expanding in the pitch dimension, and adding the feature tensor and its label to the respective lists
            data.append(torch.cat((temp_tensor, temp_tensor))[:23].unsqueeze(0))
            labels.append(key1)

    # creating the dictionary to store
    analysis_crema = {'data': data, 'labels': labels}

    # saving the .pt file
    torch.save(analysis_crema, os.path.join(base_dir, out_path))

    return labels


def create_analysis_ytrue(labels, base_dir, out_path=None):
    """
    Function for creating the ground truth file for evaluating models on the Da-TACOS analysis subset.
    The created ground truth matrix is stored as a .pt file in output_dir directory
    :param labels: labels of the files
    :param output_dir: where to store the ground truth .pt file
    """
    if out_path is None:
        out_path = 'ytrue_coveranalysis.pt'

    print(f'Creating file {out_path}...')
    ytrue = []  # empty list to store ground truth annotations
    for i in range(len(labels)):
        main_label = labels[i]  # label of the ith track in the list
        sub_ytrue = []  # empty list to store ground truth annotations for the ith track in the list
        for j in range(len(labels)):
            if labels[j] == main_label and i != j:  # checking whether the songs have the same label as the ith track
                sub_ytrue.append(1)
            else:
                sub_ytrue.append(0)
        ytrue.append(sub_ytrue)

    # saving the ground truth annotations
    torch.save(torch.Tensor(ytrue), os.path.join(base_dir, out_path))


if __name__ == '__main__':
    """
    NOTE: using option transposed_pcp=True for coveranalysis data as crema features are Tx12 there instead of 12xT
    """
    base_dir = sys.argv[1]  # path to da-tacos data
    json_path = sys.argv[2] if len(sys.argv) > 2 else None  # optional path to json file containing (split) metadata
    tag = '_' + sys.argv[3] if len(sys.argv) > 3 else None  # optional tag e.g. 'train' or 'val'
    labels = create_analysis_pt(base_dir, json_path=json_path, h5_path=None, out_path=f'coveranalysis_crema{tag}.pt', transposed_pcp=True)
    create_analysis_ytrue(labels, base_dir, out_path=f'ytrue_coveranalysis{tag}.pt')
