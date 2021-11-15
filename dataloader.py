import torch
import torch.utils.data
from torch.utils.data import Dataset
import tqdm
import random
import json
import numpy as np

from filter import bandpass_filter
from parsing.uvafdb_parser import UVAFDB_Parser
from utils.data_processing import resample_by_interpolation

DESIRED_FREQ = 360
LOW_CUT =0.33
HIGH_CUT = 50
FILTER_ORDER = 75



def shuffle_list_by_blocks(seed,list, blocksize=100):
    blocks = [list[i:i + blocksize] for i in range(0, len(list), blocksize)]
    random.Random(seed).shuffle(blocks)
    list = [elem for block in blocks for elem in block]
    return list


class ECGdataset(Dataset):
    """
    ECG dataset.
    Loads patients in Init, uses all their windows as data
    """

    def __init__(self, window_len, num_of_patients, json_filename, shuffle=False, patient_list=None):
        self.db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)
        self.all_patient_list = list(range(1, 2893))
        # patients that don't exist
        self.all_patient_list.remove(280)
        self.all_patient_list.remove(308)
        if patient_list == None:
            if shuffle:
                self.patients_to_use = random.sample(self.all_patient_list, num_of_patients)  # chooses randomly
            else:
                self.patients_to_use = self.all_patient_list[:num_of_patients]
        else:
            if shuffle:
                self.patients_to_use = random.sample(patient_list, num_of_patients)  # chooses randomly
            else:
                self.patients_to_use = patient_list[:num_of_patients]
        self.patients_to_use = [str(patient_id).zfill(4) for patient_id in self.patients_to_use]
        self.patients_data = {}
        try:
            with open(json_filename, "r") as read_file:
                dict = json.load(read_file)
        except OSError:
            print("Could not open/read " + json_filename)
        for patient_id in self.patients_to_use:
            try:
                data = dict[patient_id]
            except KeyError:
                exit(f'patient {patient_id} does not exist in pre processed Json file.')
            ecg, _ = self.db.parse_raw_ecg(patient_id=patient_id)
            # pass through filter
            ecg = bandpass_filter(ecg, id, LOW_CUT, HIGH_CUT, self.db.actual_fs, FILTER_ORDER, debug=False)
            # resample to 360hz like in paper
            ecg = resample_by_interpolation(ecg, self.db.actual_fs, DESIRED_FREQ)
            # scale ecg to be in the range [0,1]
            ecg = scale_ecg_window(ecg)
            data['ecg'] = ecg
            self.patients_data[patient_id] = data
        self.window_len = window_len
        del dict

    def __getitem__(self, idx):
        # idx = 1212144 #For overfit
        prev_id = ''
        for patient_id in sorted(self.patients_to_use):
            wanted_id = patient_id
            if self.patients_data[patient_id]['offset'] > idx:
                wanted_id = prev_id
                break
            prev_id = patient_id

        assert idx < self.patients_data[wanted_id]['offset'] + self.patients_data[wanted_id]['num_of_windows'], \
            "idx = %d < %d + %d of wanted_id = %s" % (idx, self.patients_data[wanted_id]['offset'], self.patients_data[wanted_id]['num_of_windows'], wanted_id)
        ecg = self.patients_data[wanted_id]['ecg']
        win_idx = idx - self.patients_data[wanted_id]['offset']

        start = win_idx * self.window_len
        sample = ecg[start:start + self.window_len]
        return sample

    def __len__(self):
        return len(self.patients_to_use)


class ECGdataset_oneWindowPerPatient(Dataset):
    """
    ECG dataset.
    Initializes a list of patients in init, loads patient's data in getitem and randomly chooses a window.
    """

    def __init__(self, window_len):
        self.db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)
        self.patient_list = list(range(1, 2893))
        # patients that don't exist
        self.patient_list.remove(280)
        self.patient_list.remove(308)
        self.window_len = window_len

    def __getitem__(self, idx):
        patient_id = self.patient_list[idx]
        assert patient_id > 0
        ecg, _ = self.db.parse_raw_ecg(patient_id=str(patient_id).zfill(4))
        length = len(ecg)
        num_of_windows = length // self.window_len
        idx = random.randint(0, num_of_windows)
        start = idx * self.window_len
        sample = ecg[start:start + self.window_len]

        return sample

    def __len__(self):
        return len(self.patient_list)


class ECGdataset_w_disease(Dataset):
    """
    ECG dataset.
    Loads patients in Init relevant to patient train and test lists, uses all their windows as data
    """

    def __init__(self, window_len, windows_dict, patients_list):
        self.window_len = window_len
        db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)
        self.patients_to_use = patients_list
        self.patients_to_use = [str(patient_id).zfill(4) for patient_id in self.patients_to_use]
        self.patients_data = {}

        self.windows_data = windows_dict
        for patient_id in self.patients_to_use:
            ecg, _ = db.parse_raw_ecg(patient_id=patient_id)
            # pass through filter
            ecg = bandpass_filter(ecg, id, LOW_CUT, HIGH_CUT, db.actual_fs, FILTER_ORDER, debug=False)
            # resample to 360hz like in paper
            ecg = resample_by_interpolation(ecg, db.actual_fs, DESIRED_FREQ)
            # scale ecg to be in the range [0,1]
            ecg = scale_ecg_window(ecg)
            self.patients_data[patient_id] = ecg

    def __getitem__(self, idx):
        wanted_id = self.windows_data[str(idx)]['patient']
        ecg = self.patients_data[wanted_id]
        start = self.windows_data[str(idx)]['start_idx']
        sample = ecg[start:start + self.window_len]
        return sample, idx, wanted_id, self.windows_data[str(idx)]['rlab_list']

    def __len__(self):
        return len(self.patients_to_use)


def gen_indices_by_patients(patients_list):
    try:
        with open("patient_ecg_length.json", "r") as read_file:
            dict = json.load(read_file)
    except OSError:
        print("Could not open/read patient_ecg_length.json")
    indices = []

    for patient in patients_list:
        assert patient in dict.keys()
        patient_data = dict[patient]
        patient_indices = list(
            range(patient_data['offset'], patient_data['offset'] + patient_data['num_of_windows'], 1))
        indices.extend(patient_indices)
    return indices


def gen_indices_from_lists(normal_perc, afib_perc, afib_other_perc, total_num_of_windows, filename_list,filename_data, seed=None):
    if not (0.0 <= normal_perc <= 1.0 and 0.0 <= afib_perc <= 1.0 and 0.0 <= afib_other_perc < 1.0) or normal_perc + afib_perc + afib_other_perc > 1:
        raise ValueError((normal_perc, afib_perc, afib_other_perc))

    other_perc = 1 - normal_perc - afib_perc - afib_other_perc


    normal_amount = int(total_num_of_windows * normal_perc)
    afib_amount = int(total_num_of_windows * afib_perc)
    afib_other_amount = int(total_num_of_windows * afib_other_perc)
    other_amount = int(total_num_of_windows * other_perc)
    difference = total_num_of_windows - normal_amount - afib_amount - afib_other_amount - other_amount
    if difference > 0:
        if normal_amount > 0:
            normal_amount = normal_amount + difference
        else:
            raise Exception("Need to decied what kind of windows to add")

    print(f'using windows: normal={normal_amount}, afib={afib_amount}, afib_other={afib_other_amount}, other={other_amount}')

    patients_to_load_list = []
    try:
        with open(filename_list, "r") as read_file:
            indices_list_dic = json.load(read_file)
    except OSError:
        exit(f"Could not open/read {filename_list}.")

    try:
        with open(filename_data, "r") as read_file:
            indices_data = json.load(read_file)
    except OSError:
        exit(f"Could not open/read {filename_data}.")

    shuffled_normal = shuffle_list_by_blocks(seed,indices_list_dic['normal'])
    shuffled_afib = shuffle_list_by_blocks(seed,indices_list_dic['afib'])
    shuffled_afib_and_other = shuffle_list_by_blocks(seed,indices_list_dic['afib_and_other'])
    shuffled_other = shuffle_list_by_blocks(seed,indices_list_dic['other'])

    normal_indices = shuffled_normal[:normal_amount]
    afib_indices = shuffled_afib[:afib_amount]
    afib_and_other_indices = shuffled_afib_and_other[:afib_other_amount]
    other_indices = shuffled_other[:other_amount]

    indices_list = normal_indices + afib_indices + afib_and_other_indices + other_indices
    print(f'indices_list length  = {len(indices_list)}')
    for idx in indices_list:
        if indices_data[str(idx)]['patient'] in patients_to_load_list:
            pass
        else:
            patients_to_load_list.append(indices_data[str(idx)]['patient'])
    print(f"patient to load list of len {len(patients_to_load_list)}")
    print(patients_to_load_list)
    return indices_list, patients_to_load_list


def gen_indices_by_specs(normal_perc, afib_perc, afib_other_perc, total_num_of_windows, filename):
    if not (0.0 <= normal_perc <= 1.0 and 0.0 <= afib_perc <= 1.0 and 0.0 <= afib_other_perc < 1.0) or \
            normal_perc+afib_perc+afib_other_perc > 1:
        raise ValueError((normal_perc, afib_perc, afib_other_perc))
    normal_amount = int(total_num_of_windows * normal_perc)
    afib_amount = int(total_num_of_windows * afib_perc)
    afib_other_amount = int(total_num_of_windows * afib_other_perc)
    other_amount = total_num_of_windows - normal_amount - afib_amount - afib_other_amount

    normal_idx = 0
    afib_idx =0
    afib_other_idx = 0
    other_idx = 0
    indices_list = []
    patients_to_load_list = []
    try:
        with open(filename, "r") as read_file:
            indices_data = json.load(read_file)
    except OSError:
        exit(f"Could not open/read {filename}.")

    for index in indices_data:
        if normal_idx == normal_amount and afib_idx == afib_amount and afib_other_idx == afib_other_amount and other_idx == other_amount:
            break

        if indices_data[index]['class'] == 'normal':
            if normal_idx < normal_amount:
                indices_list.append(index)
                normal_idx+=1
        elif indices_data[index]['class'] == 'afib':
            if afib_idx <afib_amount:
                indices_list.append(index)
                afib_idx+=1
        elif indices_data[index]['class'] == 'afib_and_other':
            if afib_other_idx < afib_other_amount:
                indices_list.append(index)
                afib_other_idx+=1
        elif indices_data[index]['class'] == 'other':
            if other_idx < other_amount:
                indices_list.append(index)
                other_idx+=1
        else:
            print("ERROR")
            raise ValueError(indices_data[index]['class'])

        if indices_data[index]['patient'] in patients_to_load_list:
            pass
        else:
            patients_to_load_list.append(indices_data[index]['patient'])

    if other_idx < other_amount or afib_idx < afib_amount or afib_other_idx < afib_other_amount or normal_idx < normal_amount:
        print(f'noraml_idx = {normal_idx} expected = {normal_amount}')
        print(f'afib_idx = {afib_idx} expected = {afib_amount}')
        print(f'afib_other_idx = {afib_other_idx} expected = {afib_other_amount}')
        print(f'other_idx = {other_idx} expected = {other_amount}')
        raise Exception("Did not find enough windows")
    return indices_list, patients_to_load_list


def create_train_validation_loader_for_one_patient(
        dataset: Dataset, validation_ratio, patient_id, batch_size=100, num_workers=1
):
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    from torch.utils.data import SubsetRandomSampler

    patient_indices = gen_indices_by_patients([patient_id])
    patient_indices = patient_indices[24:280]
    train_len = int((1.0 - validation_ratio) * len(patient_indices))
    random.shuffle(patient_indices)
    train_indices = patient_indices[:train_len]
    validation_indices = patient_indices[train_len:]
    print(
        f'Learning patient {patient_id}, with {len(train_indices)} windows as train and {len(validation_indices)} windows as test')

    dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                           sampler=SubsetRandomSampler(train_indices))
    dl_valid = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                           sampler=SubsetRandomSampler(validation_indices))
    return dl_train, dl_valid, patient_id, patient_id


def create_train_validation_loaders(
        dataset: Dataset, validation_ratio, patient_list, batch_size=100, train_list=None, valid_list=None,
        num_workers=1):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param patient_list: will randomize training set and test set from this list
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param valid_list: list of patients to be used instead of randomizing
    :param train_list: list of patients to be used instead of randomizing
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)
    if not ((valid_list is None and train_list is None) or (type([]) == type(valid_list) == type(train_list))):
        raise Exception("Either both or none of the lists should be passed")

    from torch.utils.data import SubsetRandomSampler

    if valid_list is None:
        print('Randomly splitting patients to train and test:')
        num_of_patients = len(patient_list)
        train_len = int((1.0 - validation_ratio) * num_of_patients)
        random.shuffle(patient_list)
        train_patients = patient_list[:train_len]
        validation_patients = patient_list[train_len:]
    else:
        print('Using saved split to train and test:')
        train_patients = train_list
        validation_patients = valid_list
    print(f'\ttrain patients ids: {train_patients}')
    print(f'\ttest patients ids: {validation_patients}')

    train_indices = gen_indices_by_patients(train_patients)
    valid_indices = gen_indices_by_patients(validation_patients)

    print(f'\ttrain number of windows: {len(train_indices)}')
    print(f'\ttest number of windows: {len(valid_indices)}')
    dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                           sampler=SubsetRandomSampler(train_indices))
    dl_valid = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                           sampler=SubsetRandomSampler(valid_indices))
    return dl_train, dl_valid, train_patients, validation_patients


def create_train_validation_loaders_w_diseases(dataset: Dataset, train_indices=None, test_indices=None, batch_size=100, num_workers=1):
    from torch.utils.data import SubsetRandomSampler
    if train_indices is not None:
        print(f'\ttrain number of windows: {len(train_indices)}')
        dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                               sampler=SubsetRandomSampler(train_indices))
    else:
        dl_train = None
    if test_indices is not None:
        print(f'\ttest number of windows: {len(test_indices)}')
        dl_valid = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                               sampler=SubsetRandomSampler(test_indices))
    else:
        dl_valid = None
    return dl_train, dl_valid


def get_window(patient_id, start=0, length=3000):
    db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)
    ecg, _ = db.parse_raw_ecg(patient_id=patient_id)  # get raw ecg
    # db.load_patient_from_disk(patient_id)
    ecg_window = ecg[start:start + length]
    frequency = db.actual_fs
    # db.plot_ecg(patient_id=patient_id,start=start,end=start+length)
    return frequency, ecg_window


def scale_ecg_window(ecg_window):
    min_val = np.min(ecg_window)
    max_val = np.max(ecg_window)

    ecg_window_scaled = ecg_window - min_val
    ecg_window_scaled = ecg_window_scaled * (1 / (max_val - min_val))

    np.any((ecg_window_scaled < 0) | (ecg_window_scaled > 1))
    return ecg_window_scaled


if __name__ == '__main__':
    # patient_list = ['1610', '1630']
    # indices = gen_indices_by_patients(patient_list)
    gen_indices_from_lists(normal_perc=0.4, afib_perc=0.2, afib_other_perc=0.4, total_num_of_windows=10, filename_list="windows_calssification_test_list.json", filename_data="windows_calssification_test.json")
    exit()
    index_list, patient_list = gen_indices_by_specs(normal_perc=0.4, afib_perc=0.2, afib_other_perc=0, total_num_of_windows=10, filename="windows_calssification_test_list.json")
    print("index list")
    print(index_list)
    print("patient_list")
    print(patient_list)
    exit()
    ecg_window = np.random.randint(8, size=(10, 1))
    ecg_window = ecg_window - 3
    print(ecg_window)
    ecg_window_scaled = scale_ecg_window(ecg_window)
    print(ecg_window_scaled)
    print('Done.')
