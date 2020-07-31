import numpy as np
from scipy import signal

B, G, R = 0, 1, 2
fs = 30
bpf_div = 60 * fs / 2
b_BPF40220, a_BPF40220 = signal.butter(10, ([40 / bpf_div, 220 / bpf_div]), 'bandpass')
mean_colors_resampled = np.zeros((3, 1))
window = 300  # Number of samples to use for every measurement
skin_vec = [0.3841, 0.5121, 0.7682]


def get_chrom_value(mean_colors_resampled, part_size):
    col_c = np.zeros((3, part_size))

    for col in [B, G, R]:
        col_stride = mean_colors_resampled[col, -part_size:]  # select last samples
        y_ACDC = signal.detrend(col_stride / np.mean(col_stride))
        col_c[col] = y_ACDC * skin_vec[col]

    x_chrom = col_c[R] - col_c[G]
    y_chrom = col_c[R] + col_c[G] - 2 * col_c[B]
    xf = bandpass_filter(x_chrom)
    yf = bandpass_filter(y_chrom)
    nx = np.std(xf)
    ny = np.std(yf)
    alpha_chrom = nx / ny

    x_stride = xf - alpha_chrom * yf
    return x_stride


def bandpass_filter(sig):
    return signal.filtfilt(b_BPF40220, a_BPF40220, sig)
