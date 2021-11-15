import os
from getpass import getuser

import h5py
import matplotlib.pyplot as plt
import json
import numpy as np
import random

from Wavelet_baseline.baseline import calc_avg_median_std
from model import *
from train_by_hdf5 import gen_dataset_from_hdf5, ECGdataset_w_disease_hdf5, create_data_loaders_hdf5
from evaluation import *
from dataloader import LOW_CUT,HIGH_CUT,FILTER_ORDER

plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

def stats(orig,reconstruct):
    rms = calc_RMS(orig, reconstruct)
    prd = calc_PRD(orig, reconstruct)
    prdn = calc_PRDN(orig, reconstruct)
    snr = calc_SNR(orig, reconstruct)
    cr = calc_CR_DEEP()
    qs = calc_QS(orig, reconstruct, cr)
    return rms,prd,prdn,snr,cr,qs


def plot_orig_and_reconstruct_inference(orig, model, str, dir='no_dir_specified', idx=None, rlab=None, save_fig=True, use_filt=False, verbose=False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        reconstruct = model(orig)
    # if use_filt:
    #     reconstruct = bandpass_filter(reconstruct, id, LOW_CUT,HIGH_CUT, db.actual_fs,FILTER_ORDER, debug=False)
    x = np.linspace(0, 5.55, num=2000)
    plt.figure(figsize=(10, 8))
    plt.plot(x,orig.cpu().detach().numpy()[0], "-b", label='Original Signal')
    plt.plot(x,reconstruct.cpu().detach().numpy()[0], "-r", label='Reconstructed Signal')
    plt.legend(loc="upper left")
    diffs = torch.abs(reconstruct[:, :-1] - reconstruct[:, 1:])
    label_diffs = torch.abs(orig[:, :-1] - orig[:, 1:])
    mask = (diffs < 0.75 * label_diffs.max(dim=1, keepdim=True)[0]).to(dtype=diffs.dtype)
    diff_masked = torch.mul(mask, diffs)
    thresh_tv = (diff_masked).sum()
    tv = diffs.sum()
    if verbose:
        print(f"orig shape = {orig.shape}, reconstruct shape = {reconstruct.shape}")
        print(f"orig: {orig.cpu().detach().squeeze().numpy()[:50]}")
        print(f"reconstruct: {reconstruct.cpu().detach().squeeze().numpy()[:50]}")
        print(f"label_diffs shape = {label_diffs.shape}, diffs shape = {diffs.shape}")
        print(f"diffs: {diffs.cpu().detach().squeeze().numpy()[:50]}")
        print(f"label_diffs: {label_diffs.cpu().detach().squeeze().numpy()[:50]}")
        print(f"mask shape = {mask.shape}, diff_masked shape = {diff_masked.shape}")
        print(f"mask: {mask.cpu().detach().squeeze().numpy()[:50]}")
        print(f"diff_masked: {diff_masked.cpu().detach().squeeze().numpy()[:50]}")
        print(f"thresh_tv shape = {thresh_tv.shape}, tv shape = {tv.shape}")
    # tv = (reconstruct[:, :-1] - reconstruct[:, 1:]).pow(2).sum()

    # plt.title(f'TV={tv}, thresh={thresh_tv}')
    plt.title(f'')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude [n.u]')
    if verbose and idx is not None:
        plt.figtext(0.5, 0.01, "index={}, rlab={}".format(idx, rlab), ha="center", fontsize=18,
                    bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    os.makedirs(dir, exist_ok=True)
    if save_fig:
        plt.savefig(os.path.join(dir, f'ecg_{str}.png'))
    plt.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reconstruct



def plot_windows_from_patients_by_windows(model, dl_test, num_of_test_idx, filename, test_num_patients, use_gpu=True, indexes_to_print=[], plot_best=False):

    plot_directory = filename.split('.')[0]
    rmss = []
    prds = []
    prdns = []
    snrs = []
    qss = []
    if plot_best:
        # MIN values
        min_rms, min_x_test,  min_idx, min_rlab = np.inf, None, -1, None
        min_prd, min_x_test_prd, min_idx_prd, min_rlab_prd = np.inf, None, -1, None
        min_prdn, min_x_test_prdn, min_idx_prdn, min_rlab_prdn = np.inf, None, -1, None
        min_snr, min_x_test_snr, min_idx_snr, min_rlab_snr = np.inf, None, -1, None
        min_qs, min_x_test_qs, min_idx_qs, min_rlab_qs = np.inf, None, -1, None
        # MAX values
        max_rms, max_x_test, max_idx, max_rlab = -np.inf, None, -1, None
        max_prd, max_x_test_prd, max_idx_prd, max_rlab_prd = -np.inf, None, -1, None
        max_prdn, max_x_test_prdn, max_idx_prdn, max_rlab_prdn = -np.inf, None, -1, None
        max_snr, max_x_test_snr, max_idx_snr, max_rlab_snr = -np.inf, None, -1, None
        max_qs, max_x_test_qs, max_idx_qs, max_rlab_qs = -np.inf, None, -1, None

    dl_iter_test = iter(dl_test)
    for i in range(num_of_test_idx):  # size of subsampler
        # plot window from test
        x_test, idx, rlab = next(dl_iter_test)

        x_test = x_test.to(torch.device('cuda' if use_gpu else 'cpu'))
        idx = idx.item()

        # rlab = [lab.item() for lab in rlab]
        reconstruct = plot_orig_and_reconstruct_inference(x_test, model, f'i_{i}_idx_{idx}', plot_directory, idx, rlab[0],
                                                          save_fig=(idx in indexes_to_print))
                                                          # save_fig=(i in [0, 100, 160, 200, 450, 493, 750]))
        rms = calc_RMS(x_test, reconstruct)
        prd = calc_PRD(x_test, reconstruct)
        prdn = calc_PRDN(x_test, reconstruct)
        snr = calc_SNR(x_test, reconstruct)
        cr = calc_CR_DEEP()
        qs = calc_QS(x_test, reconstruct,cr)

        if idx in indexes_to_print:
            print(f'idx = {idx} rms ={rms.item()} prd ={prd.item()} snr={snr.item()} qs={qs.item()}')

        rmss.append(rms.item())
        prds.append(prd.item())
        prdns.append(prdn.item())
        snrs.append(snr.item())
        qss.append(qs.item())

        if plot_best:
            if snr > 10 or snr < -5:
                plot_orig_and_reconstruct_inference(x_test, model, f'{snr.item()}_{idx}', plot_directory, idx, rlab[0],
                                                save_fig=True)
            if rms < min_rms:
                min_rms = rms
                min_x_test = x_test
                min_idx = idx
                min_rlab = rlab[0]
            if rms > max_rms:
                max_rms, max_x_test, max_idx, min_rlab = rms, x_test, idx, rlab[0]
            if prd < min_prd:
                min_prd = prd
                min_x_test_prd = x_test
                min_idx_prd = idx
                min_rlab_prd = rlab[0]
            if prd > max_prd:
                max_prd, max_x_test_prd, max_idx_prd, min_rlab_prd = rms, x_test, idx, rlab[0]
            if prdn < min_prdn:
                min_prdn = prdn
                min_x_test_prdn = x_test
                min_idx_prdn = idx
                min_rlab_prdn = rlab[0]
            if prdn > max_prdn:
                max_prdn, max_x_test_prdn, max_idx_prdn, min_rlab_prdn = rms, x_test, idx, rlab[0]
            if snr < min_snr:
                min_snr = snr
                min_x_test_snr = x_test
                min_idx_snr = idx
                min_rlab_snr = rlab[0]
            if snr > max_snr:
                max_snr, max_x_test_snr, max_idx_snr, min_rlab_snr = rms, x_test, idx, rlab[0]
            if qs < min_qs:
                min_qs = qs
                min_x_test_qs = x_test
                min_idx_qs = idx
                min_rlab_qs = rlab[0]
            if qs > max_qs:
                max_qs, max_x_test_qs, max_idx_qs, min_rlab_qs = rms, x_test, idx, rlab[0]
            # print(f'iter {i} rms = {rms}, new min:\trms{min_rms == rms},\tprd={min_prd == prd},\tprdn={min_prdn == prdn},\tsnr={min_snr == snr},\tqs={min_qs == qs}')
            print(f'iter {i}  idx {idx} rms = {rms.item()}, new min:\trms={min_rms.item() == rms.item()},\tprd={min_prd.item() == prd.item()},\tprdn={min_prdn.item() == prdn.item()},\tsnr={min_snr.item() == snr.item()},\tqs={min_qs.item() == qs.item()}')
            # print(f'\tnew max:\trms{max_rms == rms},\tprd={max_prd == prd},\tprdn={max_prdn == prdn},\tsnr={max_snr == snr},\tqs={max_qs == qs}')
            print(f'\t\tnew max:\trms{max_rms.item() == rms.item()},\tprd={max_prd.item() == prd.item()},\tprdn={max_prdn.item() == prdn.item()},\tsnr={max_snr.item() == snr.item()},\tqs={max_qs.item() == qs.item()}')

    if plot_best:
        #plot only fig with minimal and maximaml values
        plot_orig_and_reconstruct_inference(min_x_test, model, 'min_rms', plot_directory, min_idx, min_rlab, save_fig=True)
        plot_orig_and_reconstruct_inference(min_x_test_prd, model, 'min_prd', plot_directory, min_idx_prd, min_rlab_prd, save_fig=True)
        plot_orig_and_reconstruct_inference(min_x_test_prdn, model, 'min_prdn', plot_directory, min_idx_prdn, min_rlab_prdn, save_fig=True)
        plot_orig_and_reconstruct_inference(min_x_test_snr, model, 'min_snr', plot_directory, min_idx_snr, min_rlab_snr, save_fig=True)
        plot_orig_and_reconstruct_inference(min_x_test_qs, model, 'min_qs', plot_directory, min_idx_qs, min_rlab_qs, save_fig=True)

        plot_orig_and_reconstruct_inference(max_x_test, model, 'max_rms', plot_directory, max_idx, max_rlab, save_fig=True)
        plot_orig_and_reconstruct_inference(max_x_test_prd, model, 'max_prd', plot_directory, max_idx_prd, max_rlab_prd, save_fig=True)
        plot_orig_and_reconstruct_inference(max_x_test_prdn, model, 'max_prdn', plot_directory, max_idx_prdn, max_rlab_prdn, save_fig=True)
        plot_orig_and_reconstruct_inference(max_x_test_snr, model, 'max_snr', plot_directory, max_idx_snr, max_rlab_snr, save_fig=True)
        plot_orig_and_reconstruct_inference(max_x_test_qs, model, 'max_qs', plot_directory, max_idx_qs, max_rlab_qs, save_fig=True)

    rms_dic = calc_avg_median_std(rmss)
    prd_dic = calc_avg_median_std(prds)
    prdn_dic = calc_avg_median_std(prdns)
    snr_dic = calc_avg_median_std(snrs)
    qs_dic = calc_avg_median_std(qss)

    dic = {'rms': rms_dic,
           'cr': calc_CR_DEEP(),
           'prd': prd_dic,
           'prdn': prdn_dic,
           'snr': snr_dic,
           'qs': qs_dic,
           'num_of_patients': test_num_patients
           }

    with open(os.path.join(plot_directory, filename), "w") as write_file:
        json.dump(dic, write_file)

    print(f'min rms ={min(rmss)}  max rms = {max(rmss)}  avg rms = {sum(rmss) / len(rmss)}')
    return rms_dic['avg']

def get_stats_and_plot_from_list(db,index_list,model,dir,file_name):
    for idx in index_list:
        sample, _, rlab = db.__getitem__(idx)
        reconstruct  = plot_orig_and_reconstruct_inference(sample, model, f'idx_{idx}', dir='no_dir_specified', idx=idx, rlab=rlab,
                                            save_fig=True)

        rms,prd,prdn,snr,cr,qs = stats(sample, reconstruct)

        dic = {'rms': rms,
               'prd': prd,
               'prdn': prdn,
               'snr': snr,
               'qs': qs,
               'reconstruct': reconstruct}
        print(f' idx ={idx} {dic}')
        with open(os.path.join(dir, file_name), "w") as write_file:
            json.dump(dic, write_file)


# main
if __name__ == '__main__':
    # ******* PARAMS *******
    use_gpu = True
    batch_size = 1
    test_num_patients = 20
    afib_perc = 0
    sbr_perc = 0
    svta_perc = 0
    test_hdf5_fname = 'test_scaled.hdf5'

    pt_name = 'h_e91'
    output_name = '100patients'
    checkpoint_file = f'checkpoints/AutoEncoder_{pt_name}'

    # pt_name = 'h_e100'
    # output_name = 'delete'
    # checkpoint_file = f'stilted-valley-37/AutoEncoder_{pt_name}'

    # pt_name = 'h_e136'
    # output_name = 'delete'
    # checkpoint_file = f'checkpoints_feasible-breeze-43/AutoEncoder_{pt_name}'

    indexes_to_print = [1678, 1679, 960, 1138, 1218, 868, 1814]
    # indexes_to_print = []
    plot_best = False  # True
    # **********************
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Randomized a seed for a consistent list of indices")
    seed = 8415
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if os.path.isfile(f'{checkpoint_file}.pt'):

        print(f'*** Loading final checkpoint file {checkpoint_file} for inference')
    else:
        exit(f'Final checkpoint {checkpoint_file} does not exist.')

    normal_perc = 1 - afib_perc - sbr_perc - svta_perc
    data_percents = {'normal': normal_perc,
                     'afib': afib_perc,
                     'sbr': sbr_perc,
                     'svta': svta_perc}

    # Plot images from best model
    # Model
    Enc = Encoder()
    Dec = Decoder()
    model = EncoderDecoder(Enc, Dec)
    model = model.double()  # for double precision
    model.to(torch.device('cuda' if use_gpu else 'cpu'))

    saved_state = torch.load(f'/home/{getuser()}/git/ECG_compression/{checkpoint_file}.pt')  # map_location=torch.cuda.current_device())
    model.load_state_dict(saved_state['model_state'])
    print(f'Done loading /home/{getuser()}/git/ECG_compression/{checkpoint_file}.pt')

    hdf5_test_file = h5py.File(test_hdf5_fname, "r")
    test_windows, test_rlabs, amount_dic = gen_dataset_from_hdf5(afib_perc, sbr_perc, svta_perc,
                                                                                    normal_perc,
                                                                                    test_num_patients,
                                                                                    hdf5_test_file, type='test',
                                                                                    amount_dic=None)

    db = ECGdataset_w_disease_hdf5(None, None, test_windows, test_rlabs)
    # get_stats_and_plot_from_list(db, [1679,1814], model, dir,  f"inference_evaluation_{output_name}.json")
    # exit()
    _, dl_test = create_data_loaders_hdf5(db, None, len(test_rlabs), batch_size)
    num_of_test_windows = len(test_rlabs)
    plot_windows_from_patients_by_windows(model, dl_test, num_of_test_windows, f"inference_evaluation_{output_name}.json",test_num_patients, use_gpu, indexes_to_print, plot_best)

    print('Done.')



