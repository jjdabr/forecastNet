"""
Functions to generate the synthetic dataset used for the time-invariance test in section 6.1 of the ForecastNet paper
and additional helper functions relating to data formatting.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import torch

def format_input(input):
    """
    Format the input array by combining the time and input dimension of the input for feeding into ForecastNet.
    That is: reshape from [in_seq_length, n_batches, input_dim] to [n_batches, in_seq_length * input_dim]
    :param input: Input tensor with shape [in_seq_length, n_batches, input_dim]
    :return: input tensor reshaped to [n_batches, in_seq_length * input_dim]
    """
    in_seq_length, batch_size, input_dim = input.shape
    input_reshaped = input.permute(1, 0, 2)
    input_reshaped = torch.reshape(input_reshaped, (batch_size, -1))
    return input_reshaped


def batch_format(dataset, T_in_seq, T_out_seq, time_major=True):
    """
    Format the dataset into the form [T_seq, n_batches, n_dims] from the form [T, n_dims]
    :param dataset: The dataset in the form  [T, n_dims]
    :param T_in_seq: Model input sequence length
    :param T_out_seq: Model output sequence length
    :param time_major: True if the results are sent in the form [T_seq, n_batches, n_inputs]. Else results in the form
                        [n_batches, T_seq, n_inputs]
    :return: inputs: The inputs in the form [T_in_seq, n_batches, n_dims]
    :return: outputs: The inputs in the form [T_out_seq, n_batches, n_dims]
    """

    T, n_dims = dataset.shape
    inputs = []
    targets = []
    # Loop over the indexes, extract a sample at that index and run it through the model
    for t in range(T - T_in_seq - T_out_seq + 1):
        # Extract the training and testing samples at the current permuted index
        inputs.append(dataset[t: t + T_in_seq, :])
        targets.append(dataset[t + T_in_seq:t + T_in_seq + T_out_seq, :])

    # Convert lists to arrays of size [n_samples, T_in, N] and [n_samples, T_out, N]
    inputs = np.array(inputs)
    targets = np.array(targets)

    if time_major:
        inputs = np.transpose(inputs, (1, 0, 2))
        targets = np.transpose(targets, (1, 0, 2))

    return inputs, targets

def time_series(t, f=0.02):
    """
    Generate time series data over the time vector t. The value of t can be a sequence
    of integers generated using the numpy.arange() function. The default frequency is
    designed for 2750 samples.
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
    Generate a dataset using the time_series function. The function generates a dataset
    comprising 'n_seqs' time-series sequences of length T. This dataset is split into
    training, testing, and validation sets.
    returns a training,
    test, and validation dataset, each with size
    :param T: The total length of the generated time-series
    :param period: The period of the time-series seasonal component
    :param n_seqs: The number of n_seqs to generate
    :return train_data: the dataset for training the model. Shape: [n_seqs, T]
    :return test_data: the dataset for testing the model. Shape: [n_seqs, T]
    :return valid_data: the dataset for validating the model. Shape: [n_seqs, T]
    :return period: The period of the fundamental seasonal component of the time series.
    """

    # Frequency
    f = 1/period
    T_in_seq = 2 * period
    T_out_seq = period

    n_samples = T - T_in_seq - T_out_seq + 1
    test_idx = n_samples - int(0.2 * n_samples)
    valid_idx = n_samples - int(0.1 * n_samples)

    # Generate n_seqs of sequences using the time_series method
    y = []
    for i in range(n_seqs):
        idx = np.random.randint(0, T)
        y.append(time_series(np.arange(idx, idx + T), f=f))
    dataset = np.concatenate(y, axis=1)

    # Scale dataset to range [0, 1]
    minVal = 0.0
    maxVal = 1.0
    max_data_val = np.max(dataset)
    min_data_val = np.min(dataset)
    dataset = (maxVal - minVal) / (max_data_val - min_data_val) * (dataset - min_data_val) + minVal

    # Reformat dataset into batch format
    trainX_list = []
    trainY_list = []
    testX_list = []
    testY_list = []
    validX_list = []
    validY_list = []
    for i in range(n_seqs):
        # Convert to batch format
        inputs, targets = batch_format(dataset[:,[i]], T_in_seq, T_out_seq, time_major=True)
        trainX_list.append(inputs[:, :test_idx, :])
        trainY_list.append(targets[:, :test_idx, :])
        testX_list.append(inputs[:, test_idx:valid_idx, :])
        testY_list.append(targets[:, test_idx:valid_idx, :])
        validX_list.append(inputs[:, valid_idx:, :])
        validY_list.append(targets[:, valid_idx:, :])

    train_x = np.concatenate(trainX_list, axis=1)
    train_y = np.concatenate(trainY_list, axis=1)
    test_x = np.concatenate(testX_list, axis=1)
    test_y = np.concatenate(testY_list, axis=1)
    valid_x = np.concatenate(validX_list, axis=1)
    valid_y = np.concatenate(validY_list, axis=1)

    return train_x, train_y, test_x, test_y, valid_x, valid_y, period
