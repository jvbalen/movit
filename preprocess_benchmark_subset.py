import os
import sys
import json

import numpy as np
import deepdish as dd
import torch


def create_benchmark_pt(output_dir):
    """
    Function for preprocessing the cremaPCP features of the Da-TACOS benchmark subset.
    Preprocessed files are stored in output_dir directory as a single .pt file
    :param output_dir: the directory to store the .pt file
    :return: labels of the processed cremaPCP features
    """
    print('Creating benchmark_crema.pt file.')
    # reading the metadata file for the Da-TACOS benchmark subset
    with open(os.path.join(output_dir, 'da-tacos_metadata/da-tacos_benchmark_subset_metadata.json')) as f:
        metadata = json.load(f)

    data = []
    labels = []

    # iterating through the metadata file to create .pt file
    for key1 in metadata.keys():  # key1 specifies the work id
        for key2 in metadata[key1].keys():  # key2 specifies the performance id
            # loading the file
            temp_path = os.path.join(output_dir, 'da-tacos_benchmark_subset_crema/{}_crema/{}_crema.h5'.format(key1,
                                                                                                               key2))
            # reading cremaPCP features
            temp_crema = dd.io.load(temp_path)['crema']

            # downsampling the feature matrix and casting it as a torch.Tensor
            idxs = np.arange(0, temp_crema.shape[0], 8)
            temp_tensor = torch.from_numpy(temp_crema[idxs].T)

            # expanding in the pitch dimension, and adding the feature tensor and its label to the respective lists
            data.append(torch.cat((temp_tensor, temp_tensor))[:23].unsqueeze(0))
            labels.append(key1)

    # creating the dictionary to store
    benchmark_crema = {'data': data, 'labels': labels}

    # saving the .pt file
    torch.save(benchmark_crema, os.path.join(output_dir, 'benchmark_crema.pt'))

    return labels


def create_benchmark_ytrue(labels, output_dir):
    """
    Function for creating the ground truth file for evaluating models on the Da-TACOS benchmark subset.
    The created ground truth matrix is stored as a .pt file in output_dir directory
    :param labels: labels of the files
    :param output_dir: where to store the ground truth .pt file
    """
    print('Creating ytrue_benchmark.pt file.')
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
    torch.save(torch.Tensor(ytrue), os.path.join(output_dir, 'ytrue_benchmark.pt'))


if __name__ == '__main__':

    output_dir = sys.argv[1]
    labels = create_benchmark_pt(output_dir=output_dir)
    create_benchmark_ytrue(labels=labels, output_dir=output_dir)
