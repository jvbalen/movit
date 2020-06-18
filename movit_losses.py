
import gin

import torch
import torch.nn.functional as F

from utils.move_utils import pairwise_distance_matrix
from move_losses import triplet_mining_random, triplet_mining_semihard, triplet_mining_hard
from models.attention import CTransformer

@gin.configurable
def decoder_loss(seqs, labels, decoder=None, margin=1, mining_strategy=2, norm_dist=1):
    """
    Online mining function for selecting the triplets
    :param res_1: embeddings in the mini-batch
    :param move_model: model used to obtain the embeddings
    :param margin: margin for the triplet loss
    :param mining_strategy: which mining strategy to use (0 for random, 1 for semi-hard, 2 for hard)
    :param norm_dist: whether to normalize the distances by the embedding size
    :param labels: labels of the embeddings
    :return: triplet loss value
    """
    if decoder is None:
        decoder = dot_product_decoder

    # creating positive and negative masks for online mining
    aux = {}
    i_labels = []
    for l in labels:
        if l not in aux:
            aux[l] = len(aux)
        i_labels += [aux[l]]*4

    i_labels = torch.Tensor(i_labels).view(-1, 1)
    mask_diag = (1 - torch.eye(seqs.size(0))).float()
    if torch.cuda.is_available():
        i_labels = i_labels.cuda()
        mask_diag = mask_diag.cuda()
    temp_mask = (pairwise_distance_matrix(i_labels) < 0.5).float()
    mask_pos = mask_diag * temp_mask
    mask_neg = mask_diag * (1. - mask_pos)
    _, sel_pos = torch.max(mask_pos + torch.rand_like(mask_pos), 1)
    _, sel_neg = torch.max(mask_neg + torch.rand_like(mask_neg), 1)
    # print(f'sel_pos: {sel_pos}')
    # print(f'sel_neg: {sel_neg}')

    seqs_pos = torch.index_select(seqs, 0, sel_pos)
    seqs_neg = torch.index_select(seqs, 0, sel_neg)
    # print(f'seqs_pos: {seqs_pos}')
    # print(f'seqs_neg: {seqs_neg}')
    seq_pairs_pos = torch.cat((seqs.unsqueeze(1), seqs_pos.unsqueeze(1)), 1)
    seq_pairs_neg = torch.cat((seqs.unsqueeze(1), seqs_neg.unsqueeze(1)), 1)

    z_pos = decoder(seq_pairs_pos)
    z_neg = decoder(seq_pairs_neg)
    # print(f'z_pos: {z_pos}')
    # print(f'z_neg: {z_neg}')

    # calculating combined bce for pos and neg
    loss = F.binary_cross_entropy_with_logits(z_pos, torch.ones_like(z_pos)) + \
        F.binary_cross_entropy_with_logits(z_neg, torch.ones_like(z_neg))
    # loss = - z_pos + torch.log(1 + torch.exp(z_pos)) + torch.log(1 + torch.exp(z_neg))
    # print(f'loss: {loss}')

    return loss.mean()


@gin.register
def dot_product_decoder(seqs):
    """
    Compute the dot product between two sequences of length 1, stacked in dimension 1
        (batch_size, 2, 1, emb_dim) -> (batch_size,)
    """
    embeddings = seqs.squeeze(-2)
    return torch.sum(torch.prod(embeddings, -2), -1)


def self_attention_decoder(seqs, heads=2, depth=2, n_out=1):
    """
    Classify sequence pair, stacked in dimension 1, with a TransformerDecoder
        (batch_size, 2, seq_len, emb_dim) -> (batch_size,)
    """
    b, n_seq, t, e = seqs.size()
    concanated = embeddings.view(b, n_seq * t, e)
    decoder = CTransformer(e, heads, depth, seq_len, n_out)
    return decoder(concatenated)
