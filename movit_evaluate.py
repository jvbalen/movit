
import os
import json

import gin
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.move_dataset_full_size import MOVEDatasetFull
from models.move_model import MOVEModel
from models.move_model_nt import MOVEModelNT
from move_evaluate import test
from movit_train import load_data
from utils.move_utils import average_precision
from utils.move_utils import pairwise_distance_matrix
from utils.movit_utils import import_dataset_from_pt


@gin.configurable
def evaluate(Model=MOVEModel, log_dir=None,
             val_path=None, val_labels_path=None, save_metrics=True):
    """
    Main evaluation function of MOVE. For a detailed explanation of parameters
    """
    print('Evaluating model {} on dataset {}.'.format(model_name, val_path))
    # initializing the model
    # loading a pre-trained model
    # sending the model to gpu, if available
    model = Model()
    model_path = os.path.join(log_dir, 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # loading test data, initializing the dataset object and the data loader
    test_map_loader, n_test = load_data(val_path, n_bins=model.n_bins, data_aug=False, fixed_size=False)

    # calculating the pairwise distances
    # calculating the performance metrics
    dist_map_matrix = test(move_model=model,
                           test_loader=test_map_loader).cpu()
    metrics = average_precision(val_labels_path,
        -1 * dist_map_matrix.clone() + torch.diag(torch.ones(n_test) * float('-inf')))

    if save_metrics:
        with open(os.path.join(log_dir, 'eval.json'), 'w') as log:
            json.dump(metrics, log, indent='\t')

    return metrics
