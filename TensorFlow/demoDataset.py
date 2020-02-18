"""
Functions to generate the synthetic dataset used for the time-invariance test in section 6.1 of the ForecastNet paper.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""
import numpy as np

def time_series(t, f=0.02):
    """
    Generate time series data over the time vector t. The value of t can be a sequence of integers generated using the
    numpy.arange() function. The default frequency is designed for 2750 samples.
    :param t: Time vector with integer indices
    :param f: Frequency. Default is 0.02.
    :return: ys the simulated time series
    """
    T = t.size
    # Seasonal component and time-varying trend component
    ys = np.sin(2 * np.pi * f * t) * 0.6 + np.sin(1 / 5 * 2 * np.pi * f * t) * 0.2
    # Amplitude modulation component
    amp_mod = 0.5 * np.sin(1 / 6 * 2 * np.pi * f * t) + 0.8
    ys *= amp_mod
    ys = np.reshape(ys, (T,1))
    return ys

def generate_data(T = 2750, period = 50, n_seqs = 4):
    """
    Generate a dataset using the time_series function. The function generates a dataset comprising 'n_seqs' time-series
    sequences of length T. This dataset is split into training, testing, and validation sets. Returns a training,
    a test, and validation dataset.
    :param T: The total length of the generated time-series
    :param period: The period of the time-series seasonal component
    :param n_seqs: The number of n_seqs to generate
    :return train_data: the dataset for training the model. Shape: [n_seqs, T]
    :return test_data: the dataset for testing the model. Shape: [n_seqs, T]
    :return valid_data: the dataset for validating the model. Shape: [n_seqs, T]
    :return period: The period of the fundamental seasonal component of the time series.
    """

    # Use a fixed seed for repreducible results
    np.random.seed(1)

    # Frequency
    f = 1/period

    # Generate n_seqs of sequences using the time_series method
    y = []
    for i in range(n_seqs):
        idx = np.random.randint(0, T)
        y.append(time_series(np.arange(idx, idx+T), f=f))
    dataset =  np.concatenate(y, axis=1)

    # Split into a training, test, and validation sets
    test_idx = T - int(0.2 * T)
    valid_idx = T - int(0.1 * T)
    train_data = dataset[:test_idx, :].T
    test_data = dataset[test_idx:valid_idx, :].T
    valid_data = dataset[valid_idx:, :].T

    # Scale data to range [0, 1]
    minVal = 0.0
    maxVal = 1.0
    max_data_val = np.max(dataset)
    min_data_val = np.min(dataset)
    train_data = (maxVal - minVal) / (max_data_val - min_data_val) * (train_data - min_data_val) + minVal
    test_data = (maxVal - minVal) / (max_data_val - min_data_val) * (test_data - min_data_val) + minVal
    valid_data = (maxVal - minVal) / (max_data_val - min_data_val) * (valid_data - min_data_val) + minVal

    return train_data, test_data, valid_data, period
