import pickle
import pathlib
import random
import numpy as np
from SharedParams import *

amount_train = .8
amino_3grams_dict = get_amino_3grams_dict()

size = get_context_size()
path_idx = 0
for path in pathlib.Path('data').iterdir():
    if 'training' in path.name:
        print('{} ... '.format(path.name))
        with open(path, 'rb') as file:
            data_dict = pickle.load(file)
        Xs = []
        ys = []
        for key, val in data_dict.items():
            for i, char in enumerate(val):
                if i + 2 >= len(val):
                    break
                three_gram = '{}{}{}'.format(val[i], val[i+1], val[i+2])
                repr = []
                for j in range(-size * 3, 3 * size + 1, 3):
                    if j == 0:
                        continue
                    if i + j < 0 or i + j + 2 >= len(val):
                        repr.append(len(amino_3grams_dict))
                    else:
                        repr.append(amino_3grams_dict['{}{}{}'.format(val[i + j], val[i + j + 1], val[i + j + 2])])
                Xs.append(repr)
                ys.append(amino_3grams_dict[three_gram])
        np.save('data/X_{}.npy'.format(path_idx), np.array(Xs).astype(np.int))
        np.save('data/y_{}.npy'.format(path_idx), np.array(ys).astype(np.int))
        path_idx += 1
