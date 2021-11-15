import json
import random
import matplotlib.pyplot as plt
from dataloader import DESIRED_FREQ, scale_ecg_window
from filter import bandpass_filter
from parsing.uvafdb_parser import UVAFDB_Parser
import tqdm
import numpy as np

from utils.data_processing import resample_by_interpolation

from dataloader import LOW_CUT,HIGH_CUT,FILTER_ORDER
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

def plot_ecg(x,ecg,label,filename,dpi,fig_size=(10, 8)):
    plt.figure(figsize=fig_size)
    plt.plot(x,ecg, "-r", label=label)
    plt.legend(loc="upper left")
    plt.title('')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude [mV]')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


def plot_2_ecg(x,ecg1,label1,ecg2,label2,filename,dpi):
    plt.figure(figsize=(10, 8))
    plt.plot(x,ecg1, "-r", label=label1)
    plt.plot(x,ecg2, "-b", label=label2)
    plt.legend(loc="upper left")
    plt.title('')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()

def pre_proccess_signal(db, patient_id,debug = False):
    ecg, _ = db.parse_raw_ecg(patient_id=patient_id)
    window_len = 2000
    fs = 200
    if debug:
        t = np.linspace(0, len(ecg) / fs, len(ecg))
        plot_ecg(t,ecg, "Raw ECG", "pre_process_images/raw_ecg_"+patient_id+".png", 100,(20,8))
        t = np.linspace(0, window_len / fs, window_len)
        plot_ecg(t,ecg[0:window_len], "Raw ECG", "pre_process_images/raw_ecg_zoom"+patient_id+".png", 100)
    # pass through filter
    ecg = bandpass_filter(ecg, patient_id, LOW_CUT, HIGH_CUT, db.actual_fs, FILTER_ORDER, debug=debug)
    if debug:
        t = np.linspace(0, window_len / fs, window_len)
        plot_ecg(t,ecg[0:window_len], "Filtered ECG", "pre_process_images/filtered_ecg_zoom"+patient_id+".png", 100)
    # resample to 360hz like in paper
    ecg = resample_by_interpolation(ecg, db.actual_fs, DESIRED_FREQ)
    if debug:
        t = np.linspace(0, window_len / DESIRED_FREQ, window_len)
        plot_ecg(t,ecg[0:window_len], "Resampled ECG", "pre_process_images/resampled_ecg_zoom"+patient_id+".png", 100)
        scaled_ecg_for_plot = scale_ecg_window(ecg[0:window_len])
        plot_2_ecg(t,ecg[0:window_len],"Resampled ECG",scaled_ecg_for_plot, "Scaled ECG", "pre_process_images/scaled_ecg_zoom" + patient_id + ".png", 100)
    # # scale ecg to be in the range [0,1] # removed because we want to scale each window not all
    # ecg = scale_ecg_window(ecg)
    return ecg

def calc_num_of_windows_for_patient(ecg_len, window_size):
    n = ecg_len // window_size
    return n


def add_patients_to_dic(patient_id_list, dict_inst, add_labels=False):
    '''
    update dictionary to add patients that still do not exist in dictionary
    :param patient_id_list: list of patients we want to add to dictionary
    :param dict_inst: dictionary has patient_id -> ecg length,number of windows
    '''
    db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)
    for patient_id in tqdm.tqdm(patient_id_list):
        if patient_id not in dict_inst.keys():
            dict_inst[patient_id] = {}
            ecg, _ = db.parse_raw_ecg(patient_id=patient_id)
            db.load_patient_from_disk(patient_id)
            rlab = [int(num) for num in list(set(db.rlab_dict[patient_id]))]
            healthy = max(rlab) == 0.0
            dict_inst[patient_id]['length'] = len(ecg)
            dict_inst[patient_id]['rlab'] = rlab
            dict_inst[patient_id]['healthy'] = healthy


def window_calssification_to_dict(patient_id_list, dict_inst, desired_freq):
    '''
    This function creats a dictionary with
    window index : patient - the patient the window belongs to
                   start_idx - the index where the window starts after resampling to the desired frequency
                   rlab_list - the notations in the window
                   class - normal, af, af and other, other
    :param patient_id_list:
    :param dict_inst:
    :return:
    '''
    counters = {'normal': 0, 'afib': 0, 'afib_and_other': 0, 'other': 0}
    class_lists = {'normal': [], 'afib': [], 'afib_and_other': [], 'other': []}
    db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)
    window_idx = 0
    for patient_id in tqdm.tqdm(patient_id_list):
        if patient_id not in dict_inst.keys():
            ecg, frequency = db.parse_raw_ecg(patient_id=patient_id)
            ecg = resample_by_interpolation(ecg, db.actual_fs, desired_freq)
            db.load_patient_from_disk(patient_id)
            rlab = db.rlab_dict[patient_id]
            rr_sample_idx = db.rrt_dict[patient_id] * desired_freq  # [s] * [Hz] = index
            i = 0
            rr_idx = 0
            while i < len(ecg) and rr_idx < len(rr_sample_idx) and rr_idx < len(rlab):
                # check that there is a notation for this window and add it to dictionary
                if rr_sample_idx[rr_idx] <= i + 2000:
                    dict_inst[window_idx] = {}
                    dict_inst[window_idx]['patient'] = patient_id
                    dict_inst[window_idx]['start_idx'] = i
                    # find the rr labels that are relevant to this window
                    rlab_list = [rlab[rr_idx]]
                    rr_idx += 1
                    if rr_idx < len(rr_sample_idx) and rr_idx < len(rlab):
                        while rr_sample_idx[rr_idx] <= i + 2000:
                            rlab_list.append(rlab[rr_idx])
                            rr_idx +=1
                            if rr_idx == len(rr_sample_idx) or rr_idx == len(rlab):
                                break
                        rlab_list = [int(num) for num in list(set(rlab_list))]
                    dict_inst[window_idx]['rlab_list'] = rlab_list
                    classification = classify_by_rlab(rlab_list)
                    dict_inst[window_idx]['class'] = classification
                    counters[classification] += 1
                    class_lists[classification].append(window_idx)

                    window_idx += 1

                i += 2000
    print(counters)
    return class_lists


def add_window_size_to_dict(window_size, dic, max_windows=None):
    '''
    update dictionary, each patient how many windows it has in ecg of length window_size
    :param window_size: length of ecg window we want
    :param dic: dictionary has patient_id -> ecg length,number of windows

    '''
    offset = 0
    for patient_id in tqdm.tqdm(dic.keys()):
        n = calc_num_of_windows_for_patient(dic[patient_id]['length'], window_size)
        if max_windows is not None:
            windows_to_use = min(n, max_windows)
        else:
            windows_to_use = n
        dic[patient_id]['num_of_windows'] = windows_to_use
        dic[patient_id]['offset'] = offset
        offset += windows_to_use


def get_healthy_patients_list():
    with open('patient_ecg_length_100.json') as json_file:
        data = json.load(json_file)
    healthy_patients = []
    for patient in data:
        # print(data[patient]['healthy'])
        if data[patient]['healthy'] == True:
             healthy_patients.append(patient)
    return healthy_patients


def classify_by_rlab(rlab):
    '''
    Returns classification from one of four classes:
    Normal (only 0), AFIB (only 0,1),
    SBR (only 0,10), SVTA (only 0,11)
    Other else
    :param rlab: list of RR annotations from 0 to 22
    :return: class as list
    '''
    if max(rlab) == 0:
        return 'normal'
    elif max(rlab) == 1:
        return 'afib'
    elif rlab == [10] or rlab == [0,10]:
        return 'sbr'
    elif rlab == [11] or rlab == [0,11]:
        return 'svta'
    else: return 'other'



def split_patients_train_test():
    file_name = "patient_ecg_length_100.json"
    write_file_name = "patient_by_rlab.json"
    normal = []
    afib = []
    afib_and_other = []
    other = []
    try:
        with open(file_name, "r") as read_file:
            patients_dic = json.load(read_file)
    except OSError:
        exit(f"Could not open/read {file_name}.")
    for patient in patients_dic:
        labels = patients_dic[patient]['rlab']
        # print(labels)
        if max(labels) == 0:
            normal.append(patient)
        elif max(labels) == 1:
            afib.append(patient)
        elif 1 in labels:
            afib_and_other.append(patient)
        else:
            other.append(patient)
    print(f'{len(normal)} normal, {len(afib)} afib, {len(afib_and_other)} afib and other, {len(other)} other.')
    patients_rlab_dict = {'normal': normal, 'afib': afib, 'afib_and_other':afib_and_other, 'other': other}
    with open(write_file_name, "w") as write_file:
        json.dump(patients_rlab_dict, write_file)
    return patients_rlab_dict


def division_to_train_and_test(dic, test_ratio):
    write_file_name = "train_and_test_division.json"
    print('Randomly splitting patients to train and test:')
    train_patient_list = []
    test_patient_list = []
    for type in dic:
        num_of_patients = len(dic[type])
        train_len = int((1.0 - test_ratio) * num_of_patients)
        random.shuffle(dic[type])
        train_patients = dic[type][:train_len]
        test_patients = dic[type][train_len:]
        train_patient_list = train_patient_list + train_patients
        test_patient_list = test_patient_list + test_patients

    print(f'train len = {len(train_patient_list)}')
    print(f'test len = {len(test_patient_list)}')
    patients_division_dict = {'train': train_patient_list, 'test': test_patient_list}
    with open(write_file_name, "w") as write_file:
        json.dump(patients_division_dict, write_file)
    return patients_division_dict


def change_classification(dic):
    for idx in dic:
        rlab = dic[idx]['rlab_list']
        new_class = classify_by_rlab(rlab)
        dic[idx]['class'] = new_class
    return  dic


def patient_stats_and_index_list(dic, patient_list):
    p_dic = {}

    for p in patient_list:
        p_dic[p] = {}
        p_dic[p]['normal'] = 0
        p_dic[p]['afib'] = 0
        p_dic[p]['sbr'] = 0
        p_dic[p]['svta'] = 0
        p_dic[p]['other'] = 0
        p_dic[p]['normal_list'] = []
        p_dic[p]['afib_list'] = []
        p_dic[p]['sbr_list'] = []
        p_dic[p]['svta_list'] = []
        p_dic[p]['other_list'] = []


    for idx in dic:
        classification = dic[idx]['class']
        p = dic[idx]['patient']
        if classification == 'normal':
            p_dic[p]['normal']+=1
            print(idx)
            p_dic[p]['normal_list'].append(idx)
        elif classification == 'other':
            p_dic[p]['other']+=1
            p_dic[p]['other_list'].append(idx)
        elif classification == 'afib':
            p_dic[p]['afib']+=1
            p_dic[p]['afib_list'].append(idx)
        elif classification == 'sbr':
            p_dic[p]['sbr'] += 1
            p_dic[p]['sbr_list'].append(idx)
        elif classification == 'svta':
            p_dic[p]['svta'] += 1
            p_dic[p]['svta_list'].append(idx)
        else:
            raise Exception(f'unexpected classification = {classification} patient = {p} index = {idx}')
    return p_dic

if __name__ == '__main__':
    # id = ["0519","1456","2490","1536","1297", "0608","0541","2607","2841","0076", "1174","1486"]
    id = '0541'
    db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)
    ecg = pre_proccess_signal(db, id, True)
    # for i in id:
    #  ecg  = pre_proccess_signal(db, i, True)
    exit()
    dic = {}
    file_name = "windows_classification_train.json"
    file_name_list = "train_and_test_division.json"
    file_name_result = "patient_train_class_lists.json"
    try:
        with open(file_name, "r") as read_file:
            dic = json.load(read_file)
    except OSError:
        print(f"Could not open/read {file_name}. Creating new file.")

    try:
        with open(file_name_list, "r") as read_file:
            dic_list = json.load(read_file)
    except OSError:
        print(f"Could not open/read {file_name_list}. Creating new file.")

    patient_list = dic_list['train']

    res_dic = patient_stats_and_index_list(dic, patient_list)
    with open(file_name_result, "w") as write_file:
        json.dump(res_dic, write_file)

    print("Done train")
    exit()

    test_patient_list = div_dic['test']
    dic_test = {}

    class_list_test = window_calssification_to_dict(patient_id_list=test_patient_list, dict_inst=dic_test, desired_freq=360)

    with open("windows_calssification_test.json", "w") as write_file:
        json.dump(dic_test, write_file)
    with open("windows_calssification_test_list.json", "w") as write_file:
        json.dump(class_list_test, write_file)
    print("Done test")
    exit()

    file_name = "patient_ecg_length_100.json"
    patient_list = [str(num).zfill(4) for num in range(1, 11)]

    all_patient_list = list(range(1, 2893))
    # patients that don't exist
    all_patient_list.remove(280)
    all_patient_list.remove(308)
    patient_list = [str(p).zfill(4) for p in all_patient_list]

    try:
        with open(file_name, "r") as read_file:
            dic_train = json.load(read_file)
    except OSError:
        print(f"Could not open/read {file_name}. Creating new file.")
        dic_train = {}

    add_patients_to_dic(patient_id_list=patient_list, dict_inst=dic_train)
    add_window_size_to_dict(3000, dic_train, 100)

    with open(file_name, "w") as write_file:
        json.dump(dic_train, write_file)

    print('Done.')
