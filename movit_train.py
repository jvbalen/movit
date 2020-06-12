
import json
import os
import time
from collections import defaultdict

import gin
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import SGD
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.move_dataset_fixed_size import MOVEDatasetFixed
from dataset.move_dataset_full_size import MOVEDatasetFull
from models.move_model import MOVEModel
from models.move_model_nt import MOVEModelNT
from move_evaluate import test
from move_losses import triplet_loss_mining
from utils.move_utils import average_precision, triplet_mining_collate
from utils.movit_utils import import_dataset_from_pt, param_dict


@gin.configurable
def train_triplet_mining(move_model, optimizer, train_loader, margin=1.0, norm_dist=True, mining_strategy=2,
                         loss_fn=triplet_loss_mining):
    """
    Training loop for one epoch
    :param move_model: model to be trained
    :param optimizer: optimizer for training
    :param train_loader: dataloader for training
    :param margin: margin for the triplet loss
    :param norm_dist: whether to normalize distances by the embedding size
    :param mining_strategy: which online mining strategy to use
    :return: training loss of the current epoch
    """
    move_model.train()  # setting the model to training mode
    loss_log = []  # initialize the list for logging loss values of each mini-batch

    for batch in tqdm(train_loader):  # training loop
        items, labels = batch

        if torch.cuda.is_available():  # sending the pcp features and the labels to cuda if available
            items = items.cuda()

        embeddings = move_model(items)  # obtaining the embeddings of each song in the mini-batch

        # calculating the loss value of the mini-batch
        loss = loss_fn(embeddings, move_model, labels, margin=margin, mining_strategy=mining_strategy,
                       norm_dist=norm_dist)

        # setting gradients of the optimizer to zero
        optimizer.zero_grad()

        # calculating gradients with backpropagation
        loss.backward()

        # updating the weights
        optimizer.step()

        # logging the loss value of the current mini-batch
        loss_log.append(loss.cpu().item())

    train_loss = np.mean(np.array(loss_log))  # averaging the loss values of each mini-batch

    return train_loss


def validate_triplet_mining(move_model, val_loader, margin=1.0, norm_dist=True, mining_strategy=2):
    """
    validation loop for one epoch
    :param move_model: model to be used for validation
    :param val_loader: dataloader for validation
    :param margin: margin for the triplet loss
    :param norm_dist: whether to normalize distances by the embedding size
    :param mining_strategy: which online mining strategy to use
    :return: validation loss of the current epoch
    """
    with torch.no_grad():  # deactivating gradient tracking for testing
        move_model.eval()  # setting the model to evaluation mode
        loss_log = []  # initialize the list for logging loss values of each mini-batch

        for batch in tqdm(val_loader):  # training loop
            items, labels = batch

            if torch.cuda.is_available():  # sending the pcp features and the labels to cuda if available
                items = items.cuda()

            res_1 = move_model(items)  # obtaining the embeddings of each song in the mini-batch

            # calculating the loss value of the mini-batch
            loss = triplet_loss_mining(res_1, move_model, labels, margin=margin, mining_strategy=mining_strategy,
                                       norm_dist=norm_dist)

            # logging the loss value of the current mini-batch
            loss_log.append(loss.cpu().item())

        val_loss = np.mean(np.array(loss_log))  # averaging the loss values of each mini-batch

    return val_loss


@gin.configurable
def train(Model=MOVEModel, 
          train_path=None, val_path=None, val_labels_path=None, model_path=None, out_path=None,
          model_name='movit', save_model=True, save_summary=True,
          seed=42, num_of_epochs=120, trans_inv=True,
          lr=0.1, lr_phases=3, lrsch_factor=0.2, momentum=0):
    """
    Main training function of MOVE. For a detailed explanation of parameters,
    please check 'python move_main.py -- help'
    :param Model: MOVEModel or MOVEModelNT
    :param train_path: path of the training data
    :param val_path: path of the validation data
    :param val_label_path: path of the validation data labels
    :param model_name: name to save model and experiment summary
    :param save_model: whether to save model (1) or not (0)
    :param out_path: optional output path to summary (1) or not (0)
    :param seed: random seed
    :param num_of_epochs: number of epochs for training
    :param trans_inv: which model to use: MOVE (True) or MOVE without transposition invariance (False)
    :param lr: value of learning rate
    :param lr_phases: learning rate scheduler number of phases
    :param lrsch_factor: the decrease rate of learning rate
    :param momentum: momentum for optimizer
    """
    summary = defaultdict(list)  # initializing the summary dict

    # initiating the necessary random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)

    # initializing the model, gin takes care of the parameters
    model = Model()

    # sending the model to gpu, if available
    if torch.cuda.is_available():
        model.cuda()

    # initiating the optimizer
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    # load data
    train_loader, val_loader, val_map_loader, n_val = load_data(train_path, val_path, n_bins=model.n_bins)

    # initializing the learning rate scheduler
    if lr_phases > 1:
        milestones = {2: [80], 3: [80, 100]}[lr_phases]
        lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lrsch_factor)
    else:
        lr_schedule = None

    # calculating the number of parameters of the model
    n_params = sum(np.prod(p.size()) for p in model.parameters())
    print('Num of parameters = {}'.format(int(n_params)))

    print('--- Training starts ---')
    print('Model name: {}'.format(model_name))

    start_time = time.monotonic()  # start time for tracking the duration of entire training

    # main training loop
    for epoch in range(num_of_epochs):
        print(f'Epoch {epoch + 1}/{num_of_epochs}')
        last_epoch = epoch  # tracking last epoch to make sure that model didn't quit early

        start = time.monotonic()  # start time for the training loop
        train_loss = train_triplet_mining(move_model=model, optimizer=optimizer, train_loader=train_loader)
        print('Training loop: Epoch {} - Duration {:.2f} mins'.format(epoch, (time.monotonic()-start)/60))

        start = time.monotonic()  # start time for the validation loop
        val_loss = validate_triplet_mining(move_model=model, val_loader=val_loader)
        print('Validation loop: Epoch {} - Duration {:.2f} mins'.format(epoch, (time.monotonic()-start)/60))

        start = time.monotonic()  # start time for the mean average precision calculation

        # calculating the pairwise distances on validation set
        dist_map_matrix = test(move_model=model,
                               test_loader=val_map_loader).cpu()

        # calculation performance metrics
        # average_precision function uses similarities, not distances
        # we multiple the distances with -1, and set the diagonal (self-similarity) -inf
        val_map_score = average_precision(val_labels_path,
            -1 * dist_map_matrix.float().clone() + torch.diag(torch.ones(n_val) * float('-inf')))
        print('Test loop: Epoch {} - Duration {:.2f} mins'.format(epoch, (time.monotonic()-start)/60))

        # saving model if needed
        if save_model:
            if not os.path.exists('saved_models/'):
                os.mkdir('saved_models/')
            torch.save(model.state_dict(), 'saved_models/model_{}.pt'.format(model_name))

        # printing the losses
        print('training_loss: {}'.format(train_loss))
        print('val_loss: {}'.format(val_loss))
        print('val_map_score: {}'.format(val_map_score), flush=True)

        # activate learning rate scheduler if needed
        if lr_schedule:
            lr_schedule.step()

        # dumping current loss values to the summary
        summary['train_loss_log'].append(train_loss)
        summary['val_loss_log'].append(val_loss)
        summary['val_map_log'].append(val_map_score.item())

        # save summary, if needed, after each epoch
        if save_summary:
            if not os.path.exists('experiment_summaries/'):
                os.mkdir('experiment_summaries/')

            with open('experiment_summaries/summary_{}.json'.format(model_name), 'w') as log:
                json.dump(summary, log, indent='\t')

    end_time = time.monotonic()  # end time of the entire training loop

    # logging all code parameters in the summary file
    summary.update(param_dict('train'))  # adds all current params and values of the function 'train' to summary
    summary['last_epoch'] = last_epoch
    summary['training_time'] = end_time - start_time

    # saving the last version of the summary
    if save_summary:
        if not os.path.exists('experiment_summaries/'):
            os.mkdir('experiment_summaries/')

        with open('experiment_summaries/summary_{}.json'.format(model_name), 'w') as log:
            json.dump(summary, log, indent='\t')

    # saving the last version of the model
    if save_model:
        if not os.path.exists('saved_models/'):
            os.mkdir('saved_models/')
        torch.save(model.state_dict(), 'saved_models/model_{}.pt'.format(model_name))


@gin.configurable
def load_data(train_path, val_path, n_bins, patch_len=1800, data_aug=True, num_of_labels=16, ytc=False):
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
    train_data, train_labels = import_dataset_from_pt(train_path, n_bins=n_bins)
    print('Train data has been loaded!')
    val_data, val_labels = import_dataset_from_pt(val_path, n_bins=n_bins)
    print('Validation data has been loaded!')
    n_val = len(val_data)

    train_set = MOVEDatasetFixed(train_data, train_labels, h=n_bins, w=patch_len,
                                 data_aug=data_aug, ytc=ytc)
    train_loader = DataLoader(train_set, batch_size=num_of_labels, shuffle=True,
                              collate_fn=triplet_mining_collate, drop_last=True)

    val_set = MOVEDatasetFixed(val_data, val_labels, h=n_bins, w=patch_len, data_aug=0)
    val_loader = DataLoader(val_set, batch_size=num_of_labels, shuffle=True,
                            collate_fn=triplet_mining_collate, drop_last=True)

    val_map_set = MOVEDatasetFull(val_data, val_labels)
    val_map_loader = DataLoader(val_map_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, val_map_loader, n_val
