import os
import sys
import json
import random


def split_train_val(metadata, n_val=1000):
    """Utility for splitting metadata into train, validation
    """
    keys = list(metadata.keys())
    random.shuffle(keys)
    train_set = {k: metadata[k] for k in keys[:-n_val]}
    val_set = {k: metadata[k] for k in keys[-n_val:]}
    assert len(set(train_set.keys()) | set(val_set.keys())) == len(metadata)
    assert len(set(train_set.keys()) & set(val_set.keys())) == 0

    return train_set, val_set


if __name__ == '__main__':

    path = sys.argv[1]
    with open(path) as f:
        d_all = json.load(f)
        d_train, d_val = split_train_val(d_all)
    
    base, filename = os.path.split(path)
    filename, ext = os.path.splitext(filename)
    with open(os.path.join(base, f'{filename}_train_X{ext}'), 'w') as f:
        json.dump(d_train, f)
    with open(os.path.join(base, f'{filename}_val_X{ext}'), 'w') as f:
        json.dump(d_val, f)
