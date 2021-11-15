from scipy.signal import  butter, sosfiltfilt
import mne
import numpy as np
import scipy
import matplotlib.pyplot as plt

def bandpass_filter(data, id,  lowcut, highcut, signal_freq, filter_order, debug=False):
    """This function uses a Butterworth filter. The coefficoents are computed automatically. Lowcut and highcut are in Hz"""
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    sos = butter(filter_order, [low, high], btype="band", output='sos', analog=False)
    y = sosfiltfilt(sos, data)
    y = mne.filter.notch_filter(y.astype(np.float), signal_freq, freqs=60, verbose=debug)
    if debug:

        filename_spect = "exam_" + str(id) + "_spect.png"

        # get_freq_plot(data, y, sos, filter_order, signal_freq, filename_freq)
        # get_spect_plot(data, y, signal_freq, filename_spect, dpi=400)
        print("Passed first plot")
        for start in [0]:
            filename_freq = "exam_" + str(id) + "_zoom_"+str(start)+".png"
            get_spect_plot(data, y, signal_freq, filename_freq, dpi=400, num_of_samples=2000,start_sample=start, one_plot=True)

    return y

def get_spect_plot(y_orig, y_filt, fs, filename, dpi=400,num_of_samples = None,start_sample=0,one_plot = False):

    if num_of_samples != None:
        y_orig = y_orig[start_sample:start_sample+num_of_samples]
        y_filt = y_filt[start_sample:start_sample+num_of_samples]
        assert len(y_orig) == num_of_samples, f"len(y_orig) = {len(y_orig)}"
    labels = ["(a)", "(b)"]
    # get the FFT of the signals
    ps_orig = np.abs(np.fft.fft(y_orig)) ** 2
    ps_filt = np.abs(np.fft.fft(y_filt)) ** 2
    t = np.linspace(0, len(y_orig) / fs, len(y_orig))
    time_step = 1 / fs
    freqs_orig = np.fft.fftfreq(ps_orig.size, time_step)
    idx_orig = np.argsort(freqs_orig)
    freqs_filt = np.fft.fftfreq(ps_filt.size, time_step)
    idx_filt = np.argsort(freqs_filt)
    ps_orig_log = 10 * np.log10(ps_orig)
    ps_filt_log = 10 * np.log10(ps_filt)
    # plt.plot(freqs_filt[idx_filt], ps_filt_log[idx_filt], color='#3465a4', linestyle='--', label='Filtered')

    # Get the PSD of the signals using welch method
    fx, Pxx = scipy.signal.welch(y_orig, fs, nperseg=len(y_orig))
    fy, Pyy = scipy.signal.welch(y_filt, fs, nperseg=len(y_filt))
    ps_orig_log = 10 * np.log10(Pxx)
    ps_filt_log = 10 * np.log10(Pyy)
    fig = plt.figure(dpi=400)
    # plt.semilogy(fx, Pxx, label="Raw data", color='r')
    # plt.semilogy(fy, Pyy, label="Filtered", color='#3465a4', linewidth=1)
    if one_plot:
        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(t, y_orig * 10, color='r', linewidth=1,  label='Raw ECG')
        ax1.plot(t, y_filt * 10, color='#3465a4', linewidth=0.8, label='Filtered ECG')
        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel('Amplitude [mv]')
        ax1.grid(True, which='both')
        ax1.legend(loc=1)

        Y1 = ax1.get_tightbbox(fig.canvas.get_renderer())
        for a, label in zip([ax1], ""):
            bbox = a.get_tightbbox(fig.canvas.get_renderer())
            fig.text(Y1.x0 - 50, bbox.y1 + 100, labels, fontsize=14, va="top", ha="left",
                     transform=None)
    else:
        ax0 = plt.subplot(2, 1, 1)
        ax0.semilogy(fx, Pxx, color='r', linewidth=1, label='Raw ECG', )
        ax0.semilogy(fy, Pyy, color='#3465a4', linewidth=0.5, label='Filtered ECG', )

        ax0.set_title('Power Spectrum Density')
        ax0.set_xlabel('Frequency [Hz]')
        ax0.set_ylabel(r'$PSD (\frac{V^{2}}{Hz})$')
        ax0.grid(True, which='both')
        ax0.legend(loc=1)


        ax1 = plt.subplot(2, 1, 2)
        ax1.plot(t, y_orig * 10, color='r', linewidth=1)
        ax1.plot(t, y_filt * 10, color='#3465a4', linewidth=0.8)
        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel('Amplitude [mv]')
        ax1.grid(True, which='both')

        Y1 = ax0.get_tightbbox(fig.canvas.get_renderer())
        for a, label in zip([ax0, ax1], labels):
            bbox = a.get_tightbbox(fig.canvas.get_renderer())
            fig.text(Y1.x0 - 50, bbox.y1 + 100, labels, fontsize=14, va="top", ha="left",
                     transform=None)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()



if __name__ == '__main__':

    id = '1610'
    frequency, ecg_window = get_window(patient_id=id, start=2000, length=1000)
    ecg_filtered = bandpass_filter(ecg_window, id, 0.33, 50, frequency, 75, debug=True)
