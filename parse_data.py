import pathlib
import pickle

dataset_name = 'casp7'
data_dir_path = pathlib.Path(dataset_name)
if not pathlib.Path('data').exists():
    pathlib.Path('data').mkdir()

for path in data_dir_path.iterdir():
    with open(path, 'r') as file:
        print('{} ... '.format(str(path)))
        d = {}
        id_next, seq_next = False, False
        for i, line in enumerate(file.readlines()):
            if id_next:
                id = line.strip()
                id_next = False
            if seq_next:
                seq = line.strip()
                seq_next = False
                d[id] = seq
            if '[ID]' in line:
                id_next = True
            if '[PRIMARY]' in line:
                seq_next = True
        dict_path = pathlib.Path('data/dict_{}_{}.pkl'.format(dataset_name, path.name))
        with open(dict_path, 'wb') as dict_file:
            pickle.dump(d, dict_file, pickle.HIGHEST_PROTOCOL)
