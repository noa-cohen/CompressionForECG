import h5py
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset
import random
import os
import numpy as np

from model import Encoder, Decoder, EncoderDecoder
from dataloader import scale_ecg_window

num_of_windows_per_patient = 100


def create_data_loaders_hdf5(dataset: Dataset, train_num_of_indices=None, test_num_of_indices=None, batch_size=32,
                             num_workers=1):
    from torch.utils.data import SubsetRandomSampler
    from torch.utils.data import SequentialSampler

    if train_num_of_indices is not None:
        train_indices = list(range(0, train_num_of_indices))
        initial_test_index = train_num_of_indices
        print(f'\ttrain number of windows: {train_num_of_indices}')
        dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                               sampler=SubsetRandomSampler(train_indices))

    else:
        initial_test_index = 0
        dl_train = None
    if test_num_of_indices is not None:
        test_indices = list(range(initial_test_index, initial_test_index + test_num_of_indices))
        print(f'\ttest number of windows: {test_num_of_indices}')
        dl_valid = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                   sampler=SubsetRandomSampler(test_indices))


    else:
        dl_valid = None
    return dl_train, dl_valid


class ECGdataset_w_disease_hdf5(Dataset):
    """
    ECG dataset.
    """

    def __init__(self, train_windows, train_rlabs, test_windows, test_rlabs):
        if train_windows is None:
            self.windows_ecg = test_windows
            self.rlabs = test_rlabs
            print(f'len(self.windows_ecg) ={len(self.windows_ecg)}')
        else:
            self.windows_ecg = np.concatenate((train_windows, test_windows), axis=0)
            self.rlabs = train_rlabs + test_rlabs

    def __getitem__(self, idx):
        sample = self.windows_ecg[idx]
        rlab = self.rlabs[idx]
        return sample, idx, rlab

    def __len__(self):
        return len(self.windows_ecg)


def parse_data(data):
    patients = data.attrs['patients_list']
    rlab = data.attrs['rlabel']
    rlabs = [rlab] * len(patients)*100
    return data, rlabs, patients


def gen_dataset_from_hdf5(afib_perc, sbr_perc, svta_perc, norm_perc, total_num_of_patients, file, type, amount_dic):
    if not (0.0 <= norm_perc <= 1.0):
        raise ValueError((afib_perc, sbr_perc, svta_perc, norm_perc))
    total_num_of_windows = total_num_of_patients*100
    afib_amount = int(total_num_of_windows * afib_perc)
    sbr_amount = int(total_num_of_windows * sbr_perc)
    svta_amount = int(total_num_of_windows * svta_perc)
    norm_amount = int(total_num_of_windows * norm_perc)
    difference = total_num_of_windows - afib_amount - sbr_amount - svta_amount - norm_amount

    if type == 'train' or type == 'test':
        start_index_norm, start_index_afib, start_index_sbr, start_index_svta = 0, 0, 0, 0
    elif type == 'validation':
        start_index_norm = amount_dic['norm_amount']
        start_index_afib = amount_dic['afib_amount']
        start_index_sbr = amount_dic['sbr_amount']
        start_index_svta = amount_dic['svta_amount']
    else:
        raise Exception(f'Invalid type = {type}')

    if difference > 0:
        max_val = max(afib_perc, sbr_perc, svta_perc, norm_perc)
        if norm_perc == max_val:
            norm_amount += difference
        elif afib_perc == max_val:
            afib_amount += difference
        elif sbr_perc == max_val:
            sbr_amount += difference
        elif svta_perc == max_val:
            svta_amount += difference
        else:
            exit('error dividing to patients')
    print(f'using patients: normal={norm_amount}, afib={afib_amount}, sbr={sbr_amount}, svta={svta_amount}')
    print(f'start : normal={start_index_norm}, afib={start_index_afib}, sbr={start_index_sbr}, svta={start_index_svta}')

    amount_dic = {'norm_amount': norm_amount,
                  'afib_amount': afib_amount,
                  'sbr_amount': sbr_amount,
                  'svta_amount': svta_amount}

    rlabs = []
    windows = None
    if norm_amount > 0:
        data = file['normal']
        windows, rlabs, patients = parse_data(data)
        #print(f'patients = {patients}')
        #print(f'len(windows) = {len(windows)}')
        windows = windows[start_index_norm:start_index_norm+norm_amount]
        #print(f'len(windows) = {len(windows)}')
        rlabs= rlabs[start_index_norm:start_index_norm+norm_amount]
    if afib_amount > 0:
        data = file['afib']
        windows_af, rlabs_af, _ = parse_data(data)
        windows_af = windows_af[start_index_afib:start_index_afib+afib_amount]
        rlabs_af = rlabs_af[start_index_afib:start_index_afib+afib_amount]
        if windows is not None:
            windows = np.concatenate((windows, windows_af), axis=1)
        else:
            windows = windows_af
        rlabs.extend(rlabs_af)
    if sbr_amount > 0:
        data = file['sbr']
        windows_sbr, rlabs_sbr, _ = parse_data(data)
        windows_sbr = windows_sbr[start_index_sbr:start_index_sbr+sbr_amount]
        rlabs_sbr = rlabs_sbr[start_index_sbr:start_index_sbr+sbr_amount]
        if windows is not None:
            windows = np.concatenate((windows, windows_sbr), axis=1)
        else:
            windows = windows_sbr
        rlabs.extend(rlabs_sbr)
    if svta_amount > 0:
        data = file['svta']
        windows_svta, rlabs_svta, _ = parse_data(data)
        windows_svta = windows_svta[start_index_svta:start_index_svta+svta_amount]
        rlabs_svta = rlabs_svta[start_index_svta:start_index_svta+svta_amount]
        if windows is not None:
            windows = np.concatenate((windows, windows_svta), axis=1)
        else:
            windows = windows_svta
        rlabs.extend(rlabs_svta)
    print(amount_dic)
    return windows, rlabs, amount_dic

