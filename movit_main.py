import argparse
import json
import gin

from movit_train import train


if __name__:
    parser = argparse.ArgumentParser(description='Training code of MOVIT')
    parser.add_argument('-t', '--train', type=str, default=None,
                        help='Path for training data. If more than one file are used, '
                             'write only the common part')
    parser.add_argument('-v', '--val', type=str, default=None,
                        help='Path for validation data')
    parser.add_argument('-vl', '--val-labels', type=str, default=None,
                        help='Path for validation labels, formatted for ranking metrics')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Path for training data. If more than one file are used, '
                             'write only the common part')
    parser.add_argument('-o', '--out', type=str, default=None,
                        help='Path for writing output')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to gin config file')
    args = parser.parse_args()
    gin.parse_config_file(args.config)

    if args.train:
        train(train_path=args.train,
              val_path=args.val,
              val_labels_path=args.val_labels,
              model_path=args.model,
              out_path=args.out)
    else:
        raise NotImplementedError()
        # evaluate(save_name='movit',
        #          train_path=args.train_path,
        #          val_path=args.val_path,
        #          model_path=args.model_path,
        #          out_path=args.out_path)
