import h5py
import numpy as np

from dataloader import scale_ecg_window
from parsing.uvafdb_parser import UVAFDB_Parser
from pre_processing import pre_proccess_signal
import json

def create_data_for_hdf5(window_index_filename,patient_index_filename,rlab, patient_list):
    db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)
    index_list = []
    data = []
    try:
        with open(window_index_filename, "r") as read_file:
            window_index_dic = json.load(read_file)
    except OSError:
        print(f"Could not open/read {window_index_filename}. Creating new file.")

    try:
        with open(patient_index_filename, "r") as read_file:
            patient_index_dic = json.load(read_file)
    except OSError:
        print(f"Could not open/read {patient_index_filename}. Creating new file.")

    debug_counter = 0
    for p in patient_list:
        debug_counter+=1
        print(f'DEBUG{debug_counter} p = {p}')
        key = rlab + "_list"
        window_list = patient_index_dic[p][key]
        window_list = window_list[:100]

        assert len(window_list) == 100 ,'Number of windows from each window is 100'
        ecg = pre_proccess_signal(db,p)

        print('DEBUG got ECG')
        counter = 0
        for idx in window_list:
            idx_data = window_index_dic[idx]
            start = idx_data['start_idx']
            if start + 2000 < len(ecg):
                if idx_data['class'] != rlab:
                    print(f'wrong class of index = {idx}  rlab = {rlab}')
                else:
                    ecg_scaled = scale_ecg_window(ecg[start:start+2000])
                    assert ecg_scaled.min() == 0 and ecg_scaled.max() > 0.95, f"Out of range [0,1] min  ={ecg_scaled.min()}  max = {ecg_scaled.max()}"
                    data.append(ecg_scaled)
                    index_list.append(int(idx))
                    counter+=1
            else:
                print(f'bad window index = {idx}')

        assert counter == 100, 'Must add 100 windows from every patient'
    print(f' data shape = {len(data)}')
    return data, index_list

def create_hdf5_dataset(hdf5_filename, data, patients, global_idxs, rlabel:str, descr=None):
    assert len(patients) * 100 == len(global_idxs) == data.shape[0], 'wrong dimenstion'
    num_data = len(global_idxs)


    with h5py.File(hdf5_filename, 'a') as hdf_file:

        # assuming that the rank of a datum is 2 setting chunk size to match a datum
        datum_shape = data[0].shape
        chunks = (100, datum_shape[0])
        d = hdf_file.create_dataset(rlabel, data=data, chunks=chunks, maxshape=(None, 2000))
        # store some metadata
        d.attrs['rlabel'] = rlabel
        d.attrs['patients_list'] = patients


        if descr is not None:
            hdf_file.attrs['description'] = descr


        hdf_file.flush()
        hdf_file.close()

def update_global_index_file(global_index_filename,index_list,rlab):
    try:
        with open(global_index_filename, "r") as read_file:
            global_idx_dic = json.load(read_file)
    except OSError:
        print(f"Could not open/read {global_index_filename}. Creating new file.")

    if rlab in global_idx_dic.keys():
        existing = global_idx_dic[rlab]
        global_idx_dic[rlab] = existing + index_list
    else:
        global_idx_dic[rlab] = index_list

    with open(global_index_filename, "w") as write_file:
        json.dump(global_idx_dic, write_file)

def create(hdf5_filename):
    line = list(range(0, 1000))
    data = np.array([line, ] * 2000).transpose()
    patients = list(range(0, 10))
    global_idxs = list(range(0, 1000))
    rlabel = 'normal'
    create_hdf5_dataset(hdf5_filename, data, patients, global_idxs, rlabel, descr="first attempt")


def read(hdf5_filename):
    with h5py.File(hdf5_filename, 'r') as f:
        data = f['normal']
        patients = data.attrs['patients_list']
        rlab = data.attrs['rlabel']
        print(f'data[10] = {data[10]}')
        # print(f'data[5500] = {data[5500]}')


def update(hdf5_filename):
    rlab = 'normal'
    with h5py.File(hdf5_filename, 'a') as f:
        existing_data = f[rlab]
        existing_patients = existing_data.attrs['patients_list']
        existing_global_idxs = existing_data.attrs['global_idxs']
        line = list(range(0, 3000))
        data = np.array([line, ] * 2000).transpose()
        patients = list(range(0, 30))
        global_idxs = list(range(0, 3000))
        f[rlab].resize(((f[rlab].shape[0] + data.shape[0]),f[rlab].shape[1]))
        print(f'f[rlab].shape = {f[rlab].shape}')
        f[rlab][-data.shape[0]:] = data

        f[rlab].attrs['patients_list'] = np.concatenate((existing_patients, patients), axis=0)
        assert len(f[rlab].attrs['patients_list']) == 40, f'Asked for amount ={40} of patients but got different amount'
        f[rlab].attrs['global_idxs'] = np.concatenate((existing_global_idxs, global_idxs), axis=0)

def get_data_windows_from_patient(patient_list,db,window_index_dic,patient_index_dic, rlab):

    index_list = []
    data = []

    debug_counter = 0
    for p in patient_list:
        debug_counter += 1
        print(f'DEBUG{debug_counter} p = {p}')
        key = rlab + "_list"
        window_list = patient_index_dic[p][key]
        window_list = window_list[:100]

        assert len(window_list) == 100, 'Number of windows from each patient is 100'
        ecg = pre_proccess_signal(db, p)
        print('DEBUG got ECG')
        counter = 0
        for idx in window_list:
            idx_data = window_index_dic[idx]
            start = idx_data['start_idx']
            if start + 2000 < len(ecg):
                if idx_data['class'] != rlab:
                    print(f'wrong class of index = {idx}  rlab = {rlab}')
                else:
                    ecg_scaled = scale_ecg_window(ecg[start:start + 2000])
                    assert ecg_scaled.min() == 0 and ecg_scaled.max() > 0.95, f"Out of range [0,1] min  ={ecg_scaled.min()}  max = {ecg_scaled.max()}"
                    data.append(ecg_scaled)
                    index_list.append(int(idx))
                    counter += 1
            else:
                print(f'bad window index = {idx}')

        assert counter == 100, 'Must add 100 windows from every patient'
    print(f' data shape = {len(data)}')
    return np.stack(data), index_list

def add_data_to_hdf5(hdf5_filename,window_index_filename,patient_index_filename,optional_patients,patient_by_rlab_dic, rlab, amount):
    db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)

    try:
        with open(window_index_filename, "r") as read_file:
            window_index_dic = json.load(read_file)
    except OSError:
        print(f"Could not open/read {window_index_filename}. Creating new file.")

    try:
        with open(patient_index_filename, "r") as read_file:
            patient_index_dic = json.load(read_file)
    except OSError:
        print(f"Could not open/read {patient_index_filename}. Creating new file.")

    with h5py.File(hdf5_filename, 'a') as f:
        existing_data = f[rlab]
        existing_patients = existing_data.attrs['patients_list']
        amount_new_patients = amount - len(existing_patients)
        print(f'amonut of new patients = {amount_new_patients}')
        if(amount_new_patients <= 0):
            print(f'amonut of new patients = {amount_new_patients} is in valid')
            exit()
        # assert amount_new_patients > 0, f'amount of new patients to add needs to be posisitve it equals = {amount_new_patients} existing = {len(existing_patients)}'

        patients_to_add = []
        for p in optional_patients:
            if p not in existing_patients:
                print(f' {p} not in existing patients')
                # if p in patient_by_rlab_dic[rlab]:
                if patient_stats_dic[p]['afib'] < 100:
                    print('\tafib')
                    if patient_stats_dic[p]['sbr'] < 100:
                        print('\tsbr')
                        if patient_stats_dic[p]['svta'] < 100:
                            print('\tsvta')
                            print(f' {p} is best to be noraml number of windows = {patient_stats_dic[p][rlab]}')
                            if (patient_stats_dic[p][rlab] >= 100):
                                patients_to_add.append(p)
                            if len(patients_to_add) == amount_new_patients:
                                break
        print('Patients to add:')
        print(patients_to_add)
        print('Existing patients:')
        print(existing_patients)

        assert len(patients_to_add) > 0 ,f'Did not find matching patients'

        new_data, index_list = get_data_windows_from_patient(patient_list=patients_to_add,db=db,window_index_dic=window_index_dic,
                                                         patient_index_dic=patient_index_dic, rlab=rlab)

        print(f'new_data.shape =  {new_data.shape}')
        print(f' len(index_list) = {len(index_list)}')

        # Append new data to it
        f[rlab].resize(((f[rlab].shape[0] + new_data.shape[0]),f[rlab].shape[1]))
        print(f'f[rlab].shape = {f[rlab].shape}')
        f[rlab][-new_data.shape[0]:] = new_data
        existing_patients_l = existing_patients.tolist()
        f[rlab].attrs['patients_list'] = existing_patients_l + patients_to_add
        assert len(f[rlab].attrs['patients_list']) == amount, f'Asked for amount ={amount} of patients but got different amount'

        f.flush()
        f.close()

def fix_global_idx(global_index_filename):
# def fix_global_idx(hdf5_filename,global_index_filename,window_index_filename,patient_index_filename, optional_patients,patient_by_rlab_dic, rlab, amount):
    try:
        with open(global_index_filename, "r") as read_file:
            global_index_dic = json.load(read_file)
    except OSError:
        print(f"Could not open/read {global_index_filename}. Creating new file.")

    all_index = global_index_dic['global_index']


    with open(global_index_filename, "w") as write_file:
        new_global_index_dic = {'afib': all_index}
        json.dump(new_global_index_dic, write_file)
    exit()
    max_len = len(all_index)
    afib_index  = all_index[max_len-2000:max_len]
    assert len(afib_index) == 2000 , f'wrong afib length  = {len(afib_index)}'
    normal_index = all_index[:max_len-2000]
    assert len(normal_index) == 2600, f'wrong normal length  = {len(normal_index)}'
    assert len(normal_index) + len(afib_index) == len(all_index), f'total length don\'t sum up'

    new_global_index_dic = {'normal': normal_index,
                            'afib': afib_index}
    with open(global_index_filename, "w") as write_file:
        json.dump(new_global_index_dic, write_file)

    # db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)
    # try:
    #     with open(window_index_filename, "r") as read_file:
    #         window_index_dic = json.load(read_file)
    # except OSError:
    #     print(f"Could not open/read {window_index_filename}. Creating new file.")
    #
    # try:
    #     with open(patient_index_filename, "r") as read_file:
    #         patient_index_dic = json.load(read_file)
    # except OSError:
    #     print(f"Could not open/read {patient_index_filename}. Creating new file.")
    #
    # with h5py.File(hdf5_filename, 'a') as f:
    #     existing_data = f[rlab]
    #     existing_patients = existing_data.attrs['patients_list']
    #     existing_global_idxs =existing_data.attrs['global_idxs']
    #
    #
    #     patients_to_add = existing_patients[80:]
    #
    #     print(f'Patients to add: len = {len(patients_to_add)}')
    #     print(patients_to_add)
    #     print('Existing patients:')
    #     print(existing_patients)
    #
    #     assert len(patients_to_add) > 0 ,f'Did not find matching patients'
    #
    #     new_data, index_list = get_data_windows_from_patient(patient_list=patients_to_add,db=db,window_index_dic=window_index_dic,
    #                                                      patient_index_dic=patient_index_dic, rlab=rlab)
    #
    #     print(f'new_data.shape =  {new_data.shape}')
    #     print(f' len(index_list) = {len(index_list)}')
    #
    #     # Append new data to it
    #
    #     existing_global_idxs_l = existing_global_idxs.tolist()
    #     global_idxs= existing_global_idxs_l+ index_list
    #
    #     with open(global_index_filename, "w") as write_file:
    #         dic = {hdf5_filename: global_idxs}
    #         json.dump(dic, write_file)
    #
    #     f.flush()
    #     f.close()

def delete_patients(hdf5_filename, amount_patients_to_save, rlab):
    with h5py.File(hdf5_filename, 'a') as f:
        existing_data = f[rlab]
        existing_patients = existing_data.attrs['patients_list']
        existing_global_idxs =existing_data.attrs['global_idxs']

        if existing_data.shape[0] > amount_patients_to_save*100:
            existing_data = existing_data[0 - amount_patients_to_save * 100:]
            assert existing_data.shape[0] == amount_patients_to_save * 100 , f'Did not cut correctly existing_data.shape = {existing_data.shape} '
        else:
            raise Exception('Existing data is not big enough, nothing to cut')

        if len(existing_patients) > amount_patients_to_save:
            existing_patients = existing_patients[0-amount_patients_to_save]
            assert len(existing_patients) == amount_patients_to_save, f'DId not cut correctly len(existing_patients) = {len(existing_patients)}'
        else:
            print(f' Did not need to cut len(existing_patients) = {len(existing_patients)}')

        if len(existing_global_idxs) > amount_patients_to_save*100:
            existing_global_idxs = existing_global_idxs[0-amount_patients_to_save*100]
            assert len(existing_global_idxs) == existing_global_idxs*100, f'DId not cut correctly len(existing_global_idxs) = {len(existing_global_idxs)}'
        else:
            print(f' Did not need to cut len(existing_global_idxs) = {len(existing_global_idxs)}')

        #update dataset
        f[rlab] = existing_data
        f[rlab].attrs['patients_list'] = existing_patients
        f[rlab].attrs['global_idxs'] = existing_global_idxs

        f.flush()
        f.close()

def uniq_patients(hdf5_filename, potential_patients):
    exitsting_patients = []
    uniq_patients = potential_patients
    with h5py.File(hdf5_filename, 'a') as f:
        for datasets in f:  # iterate through datasets
            exitsting_patients = exitsting_patients + f[datasets].attrs['patients_list'].tolist()
            for p in potential_patients:
                if p in exitsting_patients:
                    uniq_patients.remove(p)
    return uniq_patients

if __name__ == '__main__':
    # sanity check
    # file = 'test_scaled.hdf5'
    # create(file)
    # read('test.hdf5')
    # update(file)
    # exit()

    NUM_OF_PATIENTS = 610

    rlab = 'normal'
    type = 'train'
    train_test_division_file = "train_and_test_division.json"
    patient_by_rlab_file = "patient_by_rlab.json"
    if type == 'train':
        patient_stats_file = "patient_train_stats.json"
        window_index_file = "windows_classification_train.json"
        patient_index_file = "patient_train_class_lists.json"
        hdf5_filename = 'train_scaled.hdf5'

    elif type == 'test':
        patient_stats_file = "patient_test_stats.json"
        window_index_file = "windows_classification_test.json"
        patient_index_file = "patient_test_class_lists.json"
        hdf5_filename = 'test_scaled.hdf5'

    else: raise Exception(f'Invalid type  = {type}')


    try:
        with open(train_test_division_file, "r") as read_file:
            division_dic = json.load(read_file)
    except OSError:
        print(f"Could not open/read {train_test_division_file}. Creating new file.")

    optional_patients = division_dic[type]
    try:
        with open(patient_by_rlab_file, "r") as read_file:
            patient_by_rlab_dic = json.load(read_file)
    except OSError:
        print(f"Could not open/read {patient_by_rlab_file}. Creating new file.")

    try:
        with open(patient_stats_file, "r") as read_file:
            patient_stats_dic = json.load(read_file)
    except OSError:
        print(f"Could not open/read {patient_stats_file}. Creating new file.")

    add_data_to_hdf5(hdf5_filename=hdf5_filename, window_index_filename=window_index_file, patient_index_filename=patient_index_file, optional_patients=optional_patients,
                      patient_by_rlab_dic=patient_by_rlab_dic, rlab=rlab, amount=NUM_OF_PATIENTS)
    print("Done")
    exit()


    potential_patient_list = []

    for p in optional_patients:
        # if p in patient_by_rlab_dic[rlab]:
            if (patient_stats_dic[p][rlab] >= 100):
                potential_patient_list.append(p)
            # if len(patient_list) == NUM_OF_PATIENTS:
            #     break
    print(potential_patient_list)

    del division_dic
    del patient_by_rlab_dic
    del patient_stats_dic

    uniq_patients_list = uniq_patients(hdf5_filename, potential_patient_list)

    uniq_patients_list = uniq_patients_list[0:NUM_OF_PATIENTS]
    assert len(uniq_patients_list) == NUM_OF_PATIENTS
    data, index_list = create_data_for_hdf5(window_index_filename=window_index_file,
                                            patient_index_filename=patient_index_file, rlab=rlab,
                                            patient_list=uniq_patients_list)



    create_hdf5_dataset(hdf5_filename, data=np.stack(data), patients=uniq_patients_list, global_idxs=index_list, rlabel=rlab, descr="normal train patients")



    print('Done')
