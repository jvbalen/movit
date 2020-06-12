import os
import json
import argparse

import gin

from movit_train import train
from movit_evaluate import evaluate
from utils.movit_utils import make_log_dir


if __name__:
    parser = argparse.ArgumentParser(description='Training code of MOVIT')
    parser.add_argument('-t', '--train', type=str, default=None,
                        help='Path for training data. If more than one file are used, '
                             'write only the common part')
    parser.add_argument('-v', '--val', type=str, default=None,
                        help='Path for validation data')
    parser.add_argument('-vl', '--val-labels', type=str, default=None,
                        help='Path for validation labels, formatted for ranking metrics')
    parser.add_argument('-l', '--log', type=str, default='experiments',
                        help='Path to training output (model, summaries)')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to gin config file')
    args = parser.parse_args()

    # parse config file with gin-config
    gin.parse_config_file(args.config)

    if args.train:
        log_dir = make_log_dir(args.log)
        train(train_path=args.train,
              val_path=args.val,
              val_labels_path=args.val_labels,
              log_dir=log_dir)
    else:
        evaluate(val_path=args.val,
                 val_labels_path=args.val_labels,
                 log_dir=args.log)
