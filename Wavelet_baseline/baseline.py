import json

import h5py
from pywt import wavedec
import pywt
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.signal import detrend
from train_by_hdf5 import gen_dataset_from_hdf5
from evaluation import *
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
# Orig
wave_type = 'bior4.4'
wave_level = 5

# wave_type = 'db4'
# wave_level = 6


FS = 360  # define the sampling rate

# define the number of samples of data to compress in each block
# NOTE: changing the block length affects the compression scheme for the binary map
# changing the block length means that NUM_BITS_RUN_LEN will probably have to be changed too
NUM_SAMPLES_BLOCK = 2000

# NOTE: these lengths are hard coded based on a block size of NUM_SAMPLES_BLOCK=30000
# COEFF_LENGTHS = {'cA5': 946, 'cD5': 946, 'cD4': 1883, 'cD3': 3757, 'cD2': 7506, 'cD1': 15004}
#COEFF_LENGTHS = {'cA5': 102, 'cD5': 102, 'cD4': 195, 'cD3': 382, 'cD2': 756, 'cD1': 1504}
COEFF_LENGTHS = {'cA5': 71, 'cD5': 71, 'cD4': 133, 'cD3': 257, 'cD2': 506, 'cD1': 1004}  # for win of length 2000

# number of bits that can be used to represent the run length. a 4 bit number corresponds to
# a max value of 2**4-1 = 15 bits, which is equal to 32767. so in other words, if the entire
# binary map was all 1 or all 0, NUM_BITS_RUN_LEN=4 means we can represent a run of 32767
# consecutive 0's or 1's
NUM_BITS_RUN_LEN = 4

# don't allow the PRD to be greater than 5%
MAX_PRD = 0.1

# define the threshold percentage for retaining energy of wavelet coefficients
# separate percentage for approximate coefficients and separate for detailed
THRESH_PERC_APPROX = 0.9  # 0.999
THRESH_PERC_D5 = 0.99  # 0.97
THRESH_PERC_D4_D1 = 0.9  # 0.85


def wavelet_decomposition(sig, do_plot=False):
    cA5, cD5, cD4, cD3, cD2, cD1 = wavedec(sig, 'bior4.4', level=5)
    coeffs = {'cA5': cA5, 'cD5': cD5, 'cD4': cD4, 'cD3': cD3, 'cD2': cD2, 'cD1': cD1}
    # cA6, cD6, cD5, cD4, cD3, cD2, cD1 = wavedec(sig, wave_type, level=wave_level)
    # coeffs = {'cA6': cA6, 'cD6': cD6, 'cD5': cD5, 'cD4': cD4, 'cD3': cD3, 'cD2': cD2, 'cD1': cD1}

    # plot stuff
    if do_plot:
        # print('\n\n')
        print('Plot of wavelet decomposition for all levels')
        plt.subplots(figsize=(16, 9))

        plt.subplot(6, 1, 1)
        plt.plot(coeffs['cA5'])
        plt.title('cA5')

        plt.subplot(6, 1, 2)
        plt.plot(coeffs['cD5'])
        plt.title('cD5')

        plt.subplot(6, 1, 3)
        plt.plot(coeffs['cD4'])
        plt.title('cD4')

        plt.subplot(6, 1, 4)
        plt.plot(coeffs['cD3'])
        plt.title('cD3')

        plt.subplot(6, 1, 5)
        plt.plot(coeffs['cD2'])
        plt.title('cD2')

        plt.subplot(6, 1, 6)
        plt.plot(coeffs['cD1'])
        plt.title('cD1')
        plt.xlabel('Index')

        plt.tight_layout()
        plt.savefig('figs/wavelet_decomposition.png', dpi=150)
        plt.show()

    return coeffs


def do_detrend(sig, FS, do_plot=False):
    detrended = detrend(sig)

    if do_plot:
        # print('\n\n')
        print('Original and detrended signal')

        t = [i / FS for i in range(NUM_SAMPLES_BLOCK)]
        plt.subplots(figsize=(16, 9))
        plt.plot(t, sig, label='Original Signal')
        plt.plot(t, detrended, label='Detrended Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        plt.tight_layout()
        plt.legend(loc=1)
        axes = plt.gca()
        assert (t[-1] >= 14.5)
        axes.set_xlim((10, 14.5))
        plt.savefig('figs/detrending.png', dpi=150)
        plt.show()

    return detrended


def energy(sig):
    return np.sum(sig ** 2)


def threshold_energy(coeffs, do_plot=False):
    # make a deep copy of coeffs to retain the original version
    coeffs_orig = copy.deepcopy(coeffs)

    binary_map = {}
    nonzero_coeff_count = {}

    for key in coeffs.keys():
        # sort the absolute value of the coefficients in descending order
        tmp_coeffs = np.sort(np.abs(coeffs[key]))[::-1]

        # calculate the threshold for retaining some percentage of the energy
        if key == 'cA5':
            thresh_perc = THRESH_PERC_APPROX
        elif key == 'cD5':
            thresh_perc = THRESH_PERC_D5
        else:
            thresh_perc = THRESH_PERC_D4_D1

        energy_thresholded = thresh_perc * energy(tmp_coeffs)
        energy_tmp = 0
        for coeff in tmp_coeffs:
            energy_tmp = energy_tmp + coeff ** 2

            if energy_tmp >= energy_thresholded:
                threshold = coeff
                break

        # set any coefficients below the threshold to zero
        tmp_coeffs = coeffs[key]
        inds_to_zero = np.where((tmp_coeffs < threshold) & (tmp_coeffs > -threshold))[0]
        tmp_coeffs[inds_to_zero] = 0

        # create the binary map
        binary_map_tmp = np.ones(len(coeffs[key])).astype(int)
        binary_map_tmp[inds_to_zero] = 0

        # update the various dictionaries
        coeffs[key] = tmp_coeffs
        binary_map[key] = binary_map_tmp
        nonzero_coeff_count[key] = len(tmp_coeffs)

    if do_plot:
        # print('\n\n')
        print('Plot of thresholded vs unthresholded coefficients')
        plt.subplots(figsize=(16, 9))

        plt.subplot(6, 1, 1)
        plt.plot(coeffs_orig['cA5'], label='Original')
        plt.plot(coeffs['cA5'], label='Thresholded')
        plt.legend(loc=1)
        plt.title('cA5')

        plt.subplot(6, 1, 2)
        plt.plot(coeffs_orig['cD5'], label='Original')
        plt.plot(coeffs['cD5'], label='Thresholded')
        plt.legend(loc=1)
        plt.title('cD5')

        plt.subplot(6, 1, 3)
        plt.plot(coeffs_orig['cD4'], label='Original')
        plt.plot(coeffs['cD4'], label='Thresholded')
        plt.legend(loc=1)
        plt.title('cD4')

        plt.subplot(6, 1, 4)
        plt.plot(coeffs_orig['cD3'], label='Original')
        plt.plot(coeffs['cD3'], label='Thresholded')
        plt.legend(loc=1)
        plt.title('cD3')

        plt.subplot(6, 1, 5)
        plt.plot(coeffs_orig['cD2'], label='Original')
        plt.plot(coeffs['cD2'], label='Thresholded')
        plt.legend(loc=1)
        plt.title('cD2')

        plt.subplot(6, 1, 6)
        plt.plot(coeffs_orig['cD1'], label='Original')
        plt.plot(coeffs['cD1'], label='Thresholded')
        plt.legend(loc=1)
        plt.xlabel('Index')

        plt.tight_layout()
        plt.savefig('figs/wavelet_thresholding.png', dpi=150)
        plt.show()

    return coeffs, binary_map


def scale_coeffs(coeffs, do_plot=False):
    coeffs_scaled = {}
    scaling_factors = {}

    for key in coeffs.keys():
        shift_factor = np.min(coeffs[key])
        coeffs_tmp = coeffs[key] - shift_factor

        scale_factor = np.max(coeffs_tmp)
        coeffs_tmp = coeffs_tmp / scale_factor

        scaling_factors[key] = {'shift_factor': shift_factor, 'scale_factor': scale_factor}
        coeffs_scaled[key] = coeffs_tmp

    if do_plot:
        print('\n\n')
        print('Plot of scaled coefficients:')
        plt.subplots(figsize=(16, 9))

        plt.subplot(6, 1, 1)
        plt.plot(coeffs_scaled['cA5'])
        plt.title('cA5')

        plt.subplot(6, 1, 2)
        plt.plot(coeffs_scaled['cD5'])
        plt.title('cD5')

        plt.subplot(6, 1, 3)
        plt.plot(coeffs_scaled['cD4'])
        plt.title('cD4')

        plt.subplot(6, 1, 4)
        plt.plot(coeffs_scaled['cD3'])
        plt.title('cD3')

        plt.subplot(6, 1, 5)
        plt.plot(coeffs_scaled['cD2'])
        plt.title('cD2')

        plt.subplot(6, 1, 6)
        plt.plot(coeffs_scaled['cD1'])
        plt.title('cD1')
        plt.xlabel('Index')

        plt.tight_layout()
        plt.savefig('figs/wavelet_scaled.png', dpi=150)
        plt.show()

    return coeffs_scaled, scaling_factors


def plot_wavelet_reconstruction(reconstructed, orig_data, CR, RMS, MAE, num_bits,PRD, FS, do_plot=False, num_of_coefs='', name=''):
    if do_plot:
        # print('\n\n')
        print('Plot of original signal through the process of compression and decompression')
        x = np.linspace(0, 5.55, num=2000)
        plt.figure(figsize=(10, 8))
        plt.plot(x,orig_data, "-b", label='Original Signal')
        plt.plot(x,reconstructed,  "-r", label='Reconstructed Signal')
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude [n.u.]')
        plt.legend(loc="upper left")

        #name = "CR={:.3f}, coefs={}, num_bits = {}, RMS={:.3f}, MAE={:.3f}, PRD={:.3f}, APP={}, D5={}, D1-D4={}".format(
        #    CR, num_of_coefs,num_bits, RMS,MAE,PRD,THRESH_PERC_APPROX, THRESH_PERC_D5, THRESH_PERC_D4_D1)
        plt.savefig(f'wavelet_test/reconstructed_{name}.png', dpi=150)
        plt.show()

        # # create zoomed plot
        # plt.plot(orig_data[0:750], label='Original Signal')
        # plt.plot(reconstructed[0:750], label='Reconstructed Signal')
        # # plt.title(f'CR={CR}, RMS={RMS}, APP={THRESH_PERC_APPROX}, D5={THRESH_PERC_D5}, D1-D4={THRESH_PERC_D4_D1}')
        # # plt.title("CR={:.3f}, coefs={}, num_bits = {}, RMS={:.3f}, MAE={:.3f}, PRD={:.3f}, APP={}, D5={}, D1-D4={}".format(
        # #     CR, num_of_coefs,num_bits, RMS,MAE,PRD,THRESH_PERC_APPROX, THRESH_PERC_D5, THRESH_PERC_D4_D1
        # # ))
        # # plt.title('Wavelet Reconstruction')
        # plt.xlabel('Samples')
        # plt.ylabel('Amplitude (mV)')
        # plt.ylim((-0.4, 0.4))
        # # plt.tight_layout()
        # plt.legend(loc=1)
        # axes = plt.gca()
        # # assert (t[-1] >= 14.5)
        # # axes.set_xlim((10, 14.5))
        #
        # name = "CR={:.3f}_coefs={}_RMS={:.3f}_APP={}_D5={}_D1-D4={}_ZOOM".format(
        #     CR, num_of_coefs, RMS, THRESH_PERC_APPROX, THRESH_PERC_D5, THRESH_PERC_D4_D1)
        # plt.title(name)
        # plt.savefig(f'figs/reconstructed_{i}.png', dpi=150)
        # plt.show()
        # pass


def do_quantization(coeffs, bits, do_plot=False):
    quantized_coeffs = {}

    for key in coeffs.keys():
        sig = coeffs[key]
        sig = sig * (2 ** bits - 1)
        sig = np.round(sig)
        sig = np.array(sig).astype(int)

        quantized_coeffs[key] = sig

    if do_plot:
        print('\n\n')
        print('Plot of quantized coefficients:')
        plt.subplots(figsize=(16, 9))

        plt.subplot(6, 1, 1)
        plt.plot(quantized_coeffs['cA5'])
        plt.title('cA5')

        plt.subplot(6, 1, 2)
        plt.plot(quantized_coeffs['cD5'])
        plt.title('cD5')

        plt.subplot(6, 1, 3)
        plt.plot(quantized_coeffs['cD4'])
        plt.title('cD4')

        plt.subplot(6, 1, 4)
        plt.plot(quantized_coeffs['cD3'])
        plt.title('cD3')

        plt.subplot(6, 1, 5)
        plt.plot(quantized_coeffs['cD2'])
        plt.title('cD2')

        plt.subplot(6, 1, 6)
        plt.plot(quantized_coeffs['cD1'])
        plt.title('cD1')
        plt.xlabel('Index')

        plt.tight_layout()
        plt.savefig('figs/wavelet_quantized.png', dpi=150)
        plt.show()

    return quantized_coeffs


def unscale_coeffs(coeffs_orig, coeffs_reconstructed, scaling_factors, bits, do_plot=False):
    coeffs_unscaled = {}

    for key in coeffs_reconstructed.keys():
        tmp_coeffs_unscaled = coeffs_reconstructed[key] / (2 ** bits)
        tmp_coeffs_unscaled = tmp_coeffs_unscaled * scaling_factors[key]['scale_factor']
        tmp_coeffs_unscaled = tmp_coeffs_unscaled + scaling_factors[key]['shift_factor']

        # now replace the NaN values with 0
        nan_inds = np.where(np.isnan(tmp_coeffs_unscaled))[0]
        tmp_coeffs_unscaled[nan_inds] = 0

        coeffs_unscaled[key] = tmp_coeffs_unscaled

    if do_plot:
        print('\n\n')
        print('Plot of wavelet coefficients before scaling and after rescaling:')
        plt.subplots(figsize=(16, 9))

        plt.subplot(6, 1, 1)
        plt.plot(coeffs_orig['cA5'], label='Before Scaling')
        plt.plot(coeffs_unscaled['cA5'], label='After Rescaling')
        plt.legend(loc=1)
        plt.title('cA5')

        plt.subplot(6, 1, 2)
        plt.plot(coeffs_orig['cD5'], label='Before Scaling')
        plt.plot(coeffs_unscaled['cD5'], label='After Rescaling')
        plt.legend(loc=1)
        plt.title('cD5')

        plt.subplot(6, 1, 3)
        plt.plot(coeffs_orig['cD4'], label='Before Scaling')
        plt.plot(coeffs_unscaled['cD4'], label='After Rescaling')
        plt.legend(loc=1)
        plt.title('cD4')

        plt.subplot(6, 1, 4)
        plt.plot(coeffs_orig['cD3'], label='Before Scaling')
        plt.plot(coeffs_unscaled['cD3'], label='After Rescaling')
        plt.legend(loc=1)
        plt.title('cD3')

        plt.subplot(6, 1, 5)
        plt.plot(coeffs_orig['cD2'], label='Before Scaling')
        plt.plot(coeffs_unscaled['cD2'], label='After Rescaling')
        plt.legend(loc=1)
        plt.title('cD2')

        plt.subplot(6, 1, 6)
        plt.plot(coeffs_orig['cD1'], label='Before Scaling')
        plt.plot(coeffs_unscaled['cD1'], label='After Rescaling')
        plt.legend(loc=1)
        plt.xlabel('Index')

        plt.tight_layout()
        plt.savefig('figs/wavelet_rescaled.png', dpi=150)
        plt.show()

    return coeffs_unscaled


def calculate_PRD(orig_sig, reconstructed_sig):
    num = np.sum((orig_sig - reconstructed_sig) ** 2)
    den = np.sum(orig_sig ** 2)

    PRD = np.sqrt(num / den)

    return PRD


def calculate_num_bits(orig_sig, coeffs_scaled, binary_map, scaling_factors, do_plot=False):
    # starting at 8 bits, keep decreasing the number of bits in the quantization
    # until the PRD is above some threshold
    num_bits = 9

    # initialize PRD to 0 so the while loop can run
    PRD = 0

    # keep track of PRD per number of bits
    PRD_dict = {}

    if do_plot:
        plt.subplots(figsize=(16, 9))
        t = [i / FS for i in range(NUM_SAMPLES_BLOCK)]
        plt.plot(t, orig_sig, label='Original Signal')

    while (num_bits >= 5) and (PRD <= MAX_PRD):
        # decrement the number of bits
        num_bits = num_bits - 1

        coeffs_quantized = do_quantization(coeffs_scaled, num_bits)

        # rescale the coefficients
        coeffs_unscaled = unscale_coeffs(None, coeffs_quantized, scaling_factors, num_bits)

        # do the inverse dwt
        data_reconstructed = wavelet_reconstruction(coeffs_unscaled)

        # calculate PRD
        PRD = calculate_PRD(orig_sig, data_reconstructed)
        PRD_dict[num_bits] = PRD

        # plot the reconstructed signals
        if do_plot:
            if PRD <= MAX_PRD:
                plt.plot(t, data_reconstructed, label='Reconstructed @ %i Bits, PRD = %.2f' % (num_bits, PRD))

    # if we went over the PRD, go back up by one bit
    if PRD > MAX_PRD:
        num_bits = num_bits + 1
        num_bits = min(num_bits,8)
        PRD = PRD_dict[num_bits]

    # plot some more stuff
    if do_plot:
        print('\n\n')
        print('Plots of reconstructed signals vs number of bits used for quantization:')
        plt.legend(loc=1)
        plt.tight_layout()
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        axes = plt.gca()
        axes.set_xlim((17, 21.5))
        plt.savefig('figs/PRD.png', dpi=150)
        plt.show()

    return num_bits, PRD


def combine_coefficients(coeffs, binary_map=None):
    coeffs_combined = []

    # loop through each of the wavelet decompositions and remove zero values based
    # on the binary map
    if binary_map is not None:
        for key in coeffs.keys():
            inds_to_keep = np.where(binary_map[key] == 1)[0]
            coeffs[key] = coeffs[key][inds_to_keep]

    # add in each array to coeffs_combined
    coeffs_combined.extend(coeffs['cA5'])
    coeffs_combined.extend(coeffs['cD5'])
    coeffs_combined.extend(coeffs['cD4'])
    coeffs_combined.extend(coeffs['cD3'])
    coeffs_combined.extend(coeffs['cD2'])
    coeffs_combined.extend(coeffs['cD1'])

    return coeffs_combined


def wavelet_reconstruction(coeffs):
    reconstructed = pywt.waverec([coeffs['cA5'], coeffs['cD5'], coeffs['cD4'], coeffs['cD3'],
                                  coeffs['cD2'], coeffs['cD1']], wave_type)
    # reconstructed = pywt.waverec([coeffs['cA6'], coeffs['cD6'], coeffs['cD5'], coeffs['cD4'], coeffs['cD3'], coeffs['cD2'], coeffs['cD1']], wave_type)
    return reconstructed

def compress_binary_map(binary_map):
    #define a state machine that loops through each entry in the binary map and
    #creates the compressed representation.

    #the last run count won't be included in the compressed representation, so
    #just append one more value at the end of the binary map to trigger the last
    #compression value. make a local deep copy so that the original is not affected
    binary_map = copy.deepcopy(binary_map)
    binary_map.append(int(not binary_map[-1]))


    CURRENT_STATE = binary_map[0]
    run_count = 0
    binary_string = ''

    #loop through each value in the binary map
    for val in binary_map:

        #if the current binary map value is the same as the previous one, just increment the run count
        if val == CURRENT_STATE:
            run_count = run_count + 1

        #otherwise, encode the current run count
        else:

            #handle cases where run count <= 3
            if run_count == 1:
                binary_string_tmp = '00'

            elif run_count == 2:
                binary_string_tmp = '01'

            elif run_count == 3:
                binary_string_tmp = '10'

            #otherwise, if the run count > 3
            else:
                #calculate the number bits required to represent the run count
                num_bits_run_count = len(format(run_count, 'b'))

                #build a binary string
                binary_string_tmp = ''

                #first bit represents that the run count > 3
                binary_string_tmp = binary_string_tmp + '11'

                #next 4 bits represent the number of bits that will define the run count
                binary_string_tmp = binary_string_tmp + format(num_bits_run_count, '0%ib' % NUM_BITS_RUN_LEN)

                #next number of bits is variable, and is the actual run count
                #may be up to 15 bits assuming NUM_BITS_RUN_LEN=4
                binary_string_tmp = binary_string_tmp + format(run_count, 'b')

            #print(str(run_count) + ', ' + binary_string_tmp)
            #pdb.set_trace()

            #append the binary string
            binary_string = binary_string + binary_string_tmp

            #reset the run count
            run_count = 1

        #update the current state
        CURRENT_STATE = val


    #convert the binary string into a buffer of 8 bit bytes
    byte_array = []
    for i in range(int(len(binary_string)/8)):
        byte_tmp = binary_string[i*8:(i+1)*8]
        byte_tmp = int(byte_tmp, 2)
        byte_array.append(byte_tmp)


    #check if there are any remaining bits that don't divide evenly into 8
    num_bits_last_byte = 8
    if len(binary_string)%8 != 0:
        byte_tmp = binary_string[(i+1)*8:(i+1)*8 + len(binary_string)%8]
        num_bits_last_byte = len(byte_tmp)
        byte_tmp = int(byte_tmp, 2)
        byte_array.append(byte_tmp)


    #return the initial state (ie, the first value in binary map), and the RLE binary map
    return binary_map[0], byte_array, num_bits_last_byte

def compress_coefficients(coeffs, num_bits):

    binary_string = ''

    for coeff in coeffs:
        #convert each coefficient value to binary in num_bits number of bits
        binary_string = binary_string + format(coeff, '0%ib' % num_bits)

    #loop through sets of 8 bits in the binary string and convert to a byte
    byte_array = []
    for i in range(int(len(binary_string)/8)):
        byte_tmp = binary_string[i*8:(i+1)*8]
        byte_tmp = int(byte_tmp, 2)
        byte_array.append(byte_tmp)

    #check if there are any remaining bits that don't divide evenly into 8
    #note the number of bits in this last byte for conversion back to int
    #later on
    num_bits_last_byte = 8
    if len(binary_string)%8 != 0:
        byte_tmp = binary_string[(i+1)*8:(i+1)*8 + len(binary_string)%8]
        num_bits_last_byte = len(byte_tmp)
        byte_tmp = int(byte_tmp, 2)
        byte_array.append(byte_tmp)

    return byte_array, num_bits_last_byte

def decompress_binary_map(binary_map_compressed, binary_map_initial_state, num_bits_last_byte):

    #first convert 8 bit numbers into a binary string
    binary_string = ''

    #convert each coefficient value to binary in 8 number of bits
    #note that the very last value in the the binary map may not be
    #a full 8 bits. so convert that based on num_bits_last_byte
    binary_map_len = len(binary_map_compressed)
    for i in range(binary_map_len):
        if i == binary_map_len-1:
            binary_string = binary_string + format(binary_map_compressed[i], '0%ib' % num_bits_last_byte)
        else:
            binary_string = binary_string + format(binary_map_compressed[i], '08b')


    #define a state machine that loops through each entry in the binary map and
    #creates the uncompressed representation.
    READ_HEADER = 0
    READ_NUM_BITS = 1
    READ_RUN_LEN = 2
    state = READ_HEADER

    run_type = binary_map_initial_state
    header = ''
    binary_array = np.array([])


    #loop through each value in the binary map
    for val in binary_string:

        #read the header
        if state == READ_HEADER:
            header = header + val

            if len(header) == 2:
                #run count 1
                if header == '00':
                    binary_array = np.concatenate((binary_array, np.ones(1)*run_type))
                    run_type = int(not run_type)
                    state = READ_HEADER

                #run count 2
                if header == '01':
                    binary_array = np.concatenate((binary_array, np.ones(2)*run_type))
                    run_type = int(not run_type)
                    state = READ_HEADER

                #run count 3
                if header == '10':
                    binary_array = np.concatenate((binary_array, np.ones(3)*run_type))
                    run_type = int(not run_type)
                    state = READ_HEADER

                #run count > 3
                if header == '11':
                    state = READ_NUM_BITS
                    num_bits = ''


                #reset header
                header = ''

            continue

        #read number of bits
        if state == READ_NUM_BITS:


            num_bits = num_bits + val

            if len(num_bits) == 4:
                num_bits_run_len = int(num_bits, 2)
                run_len = ''

                state = READ_RUN_LEN

            continue


        #read run length
        if state == READ_RUN_LEN:
            run_len = run_len + val

            if len(run_len) == num_bits_run_len:
                run_len = int(run_len, 2)
                binary_array = np.concatenate((binary_array, np.ones(run_len)*run_type))
                run_type = int(not run_type)
                state = READ_HEADER

            continue


    return binary_array

def decompress_coefficients(coeffs_compressed, num_bits, num_bits_last_byte):

    binary_string = ''

    #convert each coefficient value to binary in 8 number of bits
    #note that the very last value in the the binary map may not be
    #a full 8 bits. so convert that based on num_bits_last_byte
    coeffs_len = len(coeffs_compressed)
    for i in range(coeffs_len):
        if i == coeffs_len-1:
            binary_string = binary_string + format(coeffs_compressed[i], '0%ib' % num_bits_last_byte)
        else:
            binary_string = binary_string + format(coeffs_compressed[i], '08b')


    #loop through sets of num_bits bits in the binary string and convert to a byte
    byte_array = []
    for i in range(int(len(binary_string)/num_bits)):
        byte_tmp = binary_string[i*num_bits:(i+1)*num_bits]
        byte_tmp = int(byte_tmp, 2)
        byte_array.append(byte_tmp)


    return byte_array

def remap_coeffs(coeffs, binary_map):
    coeffs_remapped = np.zeros(len(binary_map))*np.nan
    inds_to_set = np.where(binary_map==1)[0]
    coeffs_remapped[inds_to_set] = coeffs

    wavelet_remapped = {}
    counter = 0
    wavelet_remapped['cA5'] = coeffs_remapped[counter:counter+COEFF_LENGTHS['cA5']]

    counter = counter + COEFF_LENGTHS['cA5']
    wavelet_remapped['cD5'] = coeffs_remapped[counter:counter+COEFF_LENGTHS['cD5']]

    counter = counter + COEFF_LENGTHS['cD5']
    wavelet_remapped['cD4'] = coeffs_remapped[counter:counter+COEFF_LENGTHS['cD4']]

    counter = counter + COEFF_LENGTHS['cD4']
    wavelet_remapped['cD3'] = coeffs_remapped[counter:counter+COEFF_LENGTHS['cD3']]

    counter = counter + COEFF_LENGTHS['cD3']
    wavelet_remapped['cD2'] = coeffs_remapped[counter:counter+COEFF_LENGTHS['cD2']]

    counter = counter + COEFF_LENGTHS['cD2']
    wavelet_remapped['cD1'] = coeffs_remapped[counter:counter+COEFF_LENGTHS['cD1']]

    return wavelet_remapped

def calculate_compression_ratio(coeffs_compressed, scaling_factors, num_bits, binary_map_compressed,
                                binary_map_initial_state):
    # each value in the compressed coefficients is 8 bits
    num_bits_compressed = len(coeffs_compressed) * 8

    # the number of bits in the last byte of the compressed coeffs is 8
    # and another 8 bits for the last byte of the compressed binary map
    num_bits_compressed = num_bits_compressed + 16

    # each set of scaling factors has 2 float values, and each float value is 32 bits
    num_bits_compressed = num_bits_compressed + len(scaling_factors) * 2 * 32

    # the number of bits corresponds to one byte
    num_bits_compressed = num_bits_compressed + 8

    # each value in the compressed binary map is 8 bits
    num_bits_compressed = num_bits_compressed + len(binary_map_compressed) * 8

    # the initial state of the binary map is just one bit but assume it's stored as a byte
    num_bits_compressed = num_bits_compressed + 8

    # each of the original data are 16 bits
    num_bits_uncompressed = NUM_SAMPLES_BLOCK * 16

    # get the compression ratio
    compression_ratio = num_bits_uncompressed / num_bits_compressed

    return compression_ratio

def calculate_rms(orig, reconstructed):
    # calculate RMS
    error = reconstructed - orig
    RMS = np.sqrt(np.mean(error.astype(float) ** 2))
    return RMS

def calculate_mae(orig, reconstructed):
    N = len(orig)
    abs_error = abs(orig-reconstructed)
    return (1/N)*sum(abs_error)

def calc_avg_median_std(all_list):
    all_list = np.array(all_list).reshape(1,-1)
    all_avg = np.mean(all_list)
    all_median = np.median(all_list)
    all_std = np.std(all_list)
    dic = {'avg': all_avg,
           'median': all_median,
           'std': all_std}
    return dic

def evaluate(fig_name='', FS=0, full_data=0):
    # load data

    # calculate the number of NUM_SAMPLES_BLOCK non-overlapping blocks of data
    N = int(len(full_data) / NUM_SAMPLES_BLOCK)

    # loop over the data in chunks of NUM_SAMPLES_BLOCK samples
    for i in range(N):
        data = full_data[i * NUM_SAMPLES_BLOCK:(i + 1) * NUM_SAMPLES_BLOCK]

        # detrend the signal as a preprocessing step to remove unecessary information
        data = do_detrend(data, FS, do_plot=False)

        # do wavelet decomposition
        coeffs = wavelet_decomposition(data)

        # threshold the coefficients such that 95% of the signal energy is retained
        # return nonzero thresholded coefficients, along with a binary map of zero/nonzero values
        # and a list of how many nonzero values were in each set of coefficients
        coeffs_thresholded, binary_map = threshold_energy(coeffs, do_plot=False)

        # do the inverse dwt
        reconstructed = wavelet_reconstruction(coeffs)

        # calculate RMS
        error = reconstructed - data
        RMS = np.sqrt(np.mean(error.astype(float) ** 2))

        # calculate CR
        num_of_coeff = 0
        for x in coeffs.values():
            num_of_coeff += np.count_nonzero(x)
        CR = NUM_SAMPLES_BLOCK / num_of_coeff
        print(f'num_of_coeff = {num_of_coeff}, CR={CR}, RMS={RMS}')

        # plot orig and reconstructed signals
        plot_wavelet_reconstruction(reconstructed, data, CR, RMS, FS, (i == 0), num_of_coeff, fig_name)

    return num_of_coeff, CR, RMS


def one_ecg_window_try_all_thresholds():
    dic = {}

    THRESH_PERC_APPROX = None
    THRESH_PERC_D5 = None
    THRESH_PERC_D4_D1 = None
    bRMS_THRESH_PERC_APPROX, bRMS_THRESH_PERC_D5, bRMS_THRESH_PERC_D4_D1 = 0, 0, 0
    bCR_THRESH_PERC_APPROX, bCR_THRESH_PERC_D5, bCR_THRESH_PERC_D4_D1 = 0, 0, 0
    min_rms = np.inf
    max_cr = -1 * np.inf
    options = [0.85, 0.9, 0.95, 0.97, 0.99]

    i = 1
    for app in options:
        for d5 in options:
            for d4d1 in options:
                plt.close('all')
                THRESH_PERC_APPROX = app
                THRESH_PERC_D5 = d5
                THRESH_PERC_D4_D1 = d4d1
                fig_name = f'_{app}_{d5}_{d4d1}'
                num_of_coeffs, cr, rms = evaluate(fig_name)
                if rms < min_rms:
                    min_rms = rms
                    bRMS_THRESH_PERC_APPROX, bRMS_THRESH_PERC_D5, bRMS_THRESH_PERC_D4_D1 = app, d5, d4d1
                if cr > max_cr:
                    max_cr = cr
                    bCR_THRESH_PERC_APPROX, bCR_THRESH_PERC_D5, bCR_THRESH_PERC_D4_D1 = app, d5, d4d1

                dic[i] = {}
                value_list = [num_of_coeffs, cr, rms, app, d5, d4d1]
                dic[i]['values'] = value_list
                i += 1

    dic[i] = {}
    value_list = [-1, -1, min_rms, bRMS_THRESH_PERC_APPROX, bRMS_THRESH_PERC_D5, bRMS_THRESH_PERC_D4_D1]
    dic[i]['values'] = value_list
    i += 1
    dic[i] = {}
    value_list = [-1, max_cr, -1, bCR_THRESH_PERC_APPROX, bCR_THRESH_PERC_D5, bCR_THRESH_PERC_D4_D1]
    dic[i]['values'] = value_list
    with open("wavelet_baseline_data.json", "w") as write_file:
        json.dump(dic, write_file)


def full_compression_reconstruction_flow(fig_name, ecg,plot_flag):
    # calculate the number of 10 second non-overlapping blocks of data
    #N = int(len(ecg) / NUM_SAMPLES_BLOCK)
    N = ecg.shape[0]
    # calculate the average CR and average PRD
    CR_avg = 0
    PRD_avg = 0

    #save values for stats
    rms_list = []
    mae_list = []
    cr_list  = []
    prd_list = []
    prdn_list = []
    snr_list = []
    qs_list = []

    # loop over the data in 10 second chunks
    for i in range(N):
        # data = ecg[i * NUM_SAMPLES_BLOCK:(i + 1) * NUM_SAMPLES_BLOCK]
        data = ecg[i]
        # print(i * NUM_SAMPLES_BLOCK)
        # print((i + 1) * NUM_SAMPLES_BLOCK)
        # detrend the signal as a preprocessing step to remove unecessary information
        print(f'max data = {max(data)} min data = {min(data)}')
        data = do_detrend(data, (i == 1))
        print(f'after deter max data = {max(data)} min data = {min(data)}')

        # do wavelet decomposition
        coeffs = wavelet_decomposition(data)

        # threshold the coefficients such that 95% of the signal energy is retained
        # return nonzero thresholded coefficients, along with a binary map of zero/nonzero values
        # and a list of how many nonzero values were in each set of coefficients
        coeffs_thresholded, binary_map = threshold_energy(coeffs, do_plot=False)#(i == 1))

        # scale each set of wavelet coefficients between zero and one
        # keep track of the scaling factors to re-scale to the original range later
        coeffs_scaled, scaling_factors = scale_coeffs(coeffs_thresholded, do_plot=False)#(i == 1))

        # quantize the coefficients. choose the number of bits to quantize based on the PRD
        num_bits, PRD_tmp = calculate_num_bits(data, coeffs_scaled, binary_map, scaling_factors, do_plot=False)#(i == 1))
        PRD_avg = PRD_avg + PRD_tmp

        # get quantized coefficients
        coeffs_quantized = do_quantization(coeffs_scaled, num_bits, do_plot=False)#(i == 1))

        # combine all the quantized coefficients into a single array for compression
        # also combine all the binary maps into a single array for compression
        coeffs_quantized_combined = combine_coefficients(coeffs_quantized, binary_map)
        binary_map_combined = combine_coefficients(binary_map)

        # compress the quantized coefficients
        coeffs_quantized_compressed, num_bits_last_byte_coeffs = compress_coefficients(coeffs_quantized_combined,
                                                                                       num_bits)

        # compress the binary map
        binary_map_initial_state, binary_map_compressed, num_bits_last_byte_binary_map = compress_binary_map(
            binary_map_combined)

        # "transmit" all the necessary information to reconstruct the signal on the recieving end
        # this includes:
        # 1. the compressed coefficients
        # 2. the number of bits associated with the last byte of the compressed coefficients
        # 3. the scaling factors for each wavelet decomposition
        # 4. the number of bits used to quantize the coefficients
        # 5. the compressed binary map
        # 6. the number of bits associated with the last byte of the compressed binary map
        # 7. the initial state of the binary map

        # calculate the compression ratio for this transmission
        CR_tmp = calculate_compression_ratio(coeffs_quantized_compressed, scaling_factors, num_bits,
                                             binary_map_compressed, binary_map_initial_state)
        CR_avg = CR_avg + CR_tmp

        # decompress the binary map
        binary_map_decompressed = decompress_binary_map(binary_map_compressed, binary_map_initial_state,
                                                        num_bits_last_byte_binary_map)

        # decompress the coefficients
        coeffs_decompressed = decompress_coefficients(coeffs_quantized_compressed, num_bits, num_bits_last_byte_coeffs)

        # remap all the coefficients back to their original wavelet decompositions
        coeffs_reconstructed = remap_coeffs(coeffs_decompressed, binary_map_decompressed)

        # rescale the coefficients
        coeffs_unscaled = unscale_coeffs(coeffs, coeffs_reconstructed, scaling_factors, num_bits, do_plot=False)#(i == 1))

        # do the inverse dwt
        data_reconstructed = wavelet_reconstruction(coeffs_unscaled)

        data_for_stats = torch.from_numpy(data).unsqueeze(0)
        data_reconstructed_for_stats = torch.from_numpy(data_reconstructed).unsqueeze(0)
        print(f'max data_for_stats = {max(data_for_stats)} min data_for_stats = {min(data_for_stats)}')

        #Calculate root mean square error
        rms = calc_RMS(data_for_stats, data_reconstructed_for_stats)

        #Calculate absolute mean error
        mae = calculate_mae(data,data_reconstructed)

        prd_tmpp = calc_PRD(data_for_stats, data_reconstructed_for_stats)
        prdn_tmp = calc_PRDN(data_for_stats, data_reconstructed_for_stats)
        snr_tmp = calc_SNR(data_for_stats, data_reconstructed_for_stats)
        qs_tmp = calc_QS(data_for_stats, data_reconstructed_for_stats, CR_tmp)

        num_of_coeff = 0
        for x in coeffs.values():
            num_of_coeff += np.count_nonzero(x)

        #add stas to list
        rms_list.append(rms)
        mae_list.append(mae)
        cr_list.append(CR_tmp)
        prd_list.append(prd_tmpp)
        prdn_list.append(prdn_tmp)
        snr_list.append(snr_tmp)
        qs_list.append(qs_tmp)

        # print(f'{i}: rms = {rms.item()} mae = {mae}  num_bits ={num_bits} num_of_coeffs = {num_of_coeff}  CR = {CR_tmp} THRESH_PERC_APPROX={THRESH_PERC_APPROX} THRESH_PERC_D5={THRESH_PERC_D5} THRESH_PERC_D4_D1={THRESH_PERC_D4_D1}')
        if i in [1679, 1678, 960, 1138, 1218, 868]:
            print(f'**{i}: RMS={rms.item()}, PRD={prd_tmpp.item()}, SNR={snr_tmp.item()}, QS={qs_tmp.item()}, CR={CR_tmp}')

        if plot_flag:
            plot_wavelet_reconstruction(data_reconstructed, data, CR_tmp, rms, mae, num_bits,PRD_tmp, FS, (i in [1679,1678,960,1138,1218,868]), num_of_coeff, str(i))


    CR_avg = CR_avg / N
    PRD_avg = PRD_avg / N
    print('Average compression ratio: %.1f' % CR_avg)
    print('Average PRD: %.3f' % PRD_avg)
    return rms_list, mae_list, cr_list, prd_list , prdn_list, snr_list, qs_list


def stats_full_c_r_flow(test_hdf5_fname,data_fname,afib_perc, sbr_perc, svta_perc, normal_perc,
                                                                 test_num_patients):

    rms_all = []
    mae_all = []
    cr_all = []
    prd_all = []
    prdn_all = []
    snr_all = []
    qs_all = []

    hdf5_test_file = h5py.File(test_hdf5_fname, "r")
    test_windows, test_rlabs, amount_dic = gen_dataset_from_hdf5(afib_perc, sbr_perc, svta_perc,
                                                                 normal_perc,
                                                                 test_num_patients,
                                                                 hdf5_test_file, type='test',
                                                                 amount_dic=None)


    #compress and reconstruct
    rms_list, mae_list, cr_list, prd_list, prdn_list, snr_list, qs_list=full_compression_reconstruction_flow(id, test_windows, plot_flag=True)
    rms_all.append(rms_list)
    mae_all.append(mae_list)
    cr_all.append(cr_list)
    prd_all.append(prd_list)
    prdn_all.append(prdn_list)
    snr_all.append(snr_list)
    qs_all.append(qs_list)

    rms_dic  = calc_avg_median_std(rms_all)
    mae_dic  = calc_avg_median_std(mae_all)
    cr_dic   = calc_avg_median_std(cr_all)
    prd_dic  = calc_avg_median_std(prd_all)
    prdn_dic = calc_avg_median_std(prdn_all)
    snr_dic  = calc_avg_median_std(snr_all)
    qs_dic   = calc_avg_median_std(qs_all)

    dic = {'rms': rms_dic,
           'mae': mae_dic,
           'cr': cr_dic,
           'prd': prd_dic,
           'prdn': prdn_dic,
           'snr': snr_dic,
           'qs': qs_dic
    }

    with open(data_fname, "w") as write_file:
        json.dump(dic, write_file)
    print('Average RMS: %.3f' % rms_dic['avg'])


if __name__ == '__main__':
    dic = {}

    # # Thersholds from paper
    # THRESH_PERC_APPROX = 0.999
    # THRESH_PERC_D5 = 0.97
    # THRESH_PERC_D4_D1 = 0.85
    # # # MAX_PRD = 0.4
    # MAX_PRD = 0.12

    # Thresholds for CR of 32.3
    THRESH_PERC_APPROX = 0.96
    THRESH_PERC_D5 = 0.2
    THRESH_PERC_D4_D1 = 0.1
    MAX_PRD = 0.4

    plt.close('all')

    stats_full_c_r_flow(test_hdf5_fname='../test_scaled.hdf5',data_fname = 'wavelet_test/baseline.json',afib_perc=0,
                        sbr_perc=0, svta_perc=0, normal_perc=1, test_num_patients=20)

