"""
Helper functions to calculate error metrics.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import warnings

def calculate_error(Yhat, Y, print_errors=False):
    """
    Calculate the Mean Absolute Scaled Error (MASE) and the Symmetric Mean Absolute
    Percentage Error (SMAPE) on a forecast Yhat given the target Y.
    Both Yhat and Y can be in one of the following forms:
    * One dimensional arrays
    * Two dimensional arrays with several sequences along the first dimension (dimension 0).
    * Three dimensional arrays with several sequences along first dimension (dimension 0) and with the third dimension
      (dimension 2) being of size 1.
    :param Yhat: The forecast
    :param Y: The target
    :return: mase: Mean Absolute Scaled Error (MASE)
    :return: smape: Symmetric Mean Absolute Percentage Error (SMAPE)
    """

    # Ensure arrays are 2D
    assert np.ndim(Y) <= 3, 'Y must be one, two, or three dimensional, with the sequence on the first dimension'
    assert np.ndim(Yhat) <= 3, 'Yhat must be one, two, or three dimensional, with the sequence on the first dimension'
    assert np.ndim(Y) <= np.ndim(Yhat), 'Y has a different shape to Yhat'

    # Prepare Y and Yhat based on their number of dimensions
    if np.ndim(Y) == 1:
        n_sequences = 1
        Y = np.expand_dims(Y, axis=1)
        Yhat = np.expand_dims(Yhat, axis=1)
    elif np.ndim(Y) == 2:
        n_sequences = Y.shape[1]
    elif np.ndim(Y) == 3:
        assert Y.shape[2] == 1, 'For a three dimensional array, Y.shape[2] == 1'
        Y = np.squeeze(Y, axis=2)
        assert Yhat.shape[2] == 1, 'For a three dimensional array, Y.shape[2] == 1'
        Yhat = np.squeeze(Yhat, axis=2)
        n_sequences = Y.shape[1]
    else:
        raise Warning('Error in dimensions')


    # Symmetric Mean Absolute Percentage Error (M4 comp)
    smape = []
    for i in range(n_sequences):
        # Compute numerator and denominator
        numerator = np.absolute(Y[:, i] - Yhat[:, i])
        denominator = (np.absolute(Y[:, i]) + np.absolute(Yhat[:, i]))
        # Remove any elements with zeros in the denominator
        non_zeros = denominator != 0
        numerator = numerator[non_zeros]
        denominator = denominator[non_zeros]
        # Sequence length
        length = numerator.shape[0]
        # Calculate error
        smape.append(200.0 / length * np.sum(numerator / denominator))
    smape = np.array(smape)
    if print_errors:
        print('Symmetric mean absolute percentage error (sMAPE) = ', smape)

    # Mean absolute scaled error
    se = []
    mase = []
    for i in range(n_sequences):
        numerator = (Y[:, i] - Yhat[:, i])
        denominator = np.sum(np.absolute(Y[1:, i] - Y[0:-1, i]), axis=0)
        # Check if denominator is zero
        if denominator == 0:
            warnings.warn("The denominator for the MASE is zero")
            se.append(np.NaN * np.ones(length))
            mase.append(np.NaN)
            continue
        # Sequence length
        length = numerator.shape[0]
        # Scaled Error
        scaled_error = (length - 1) * numerator / denominator
        se.append(scaled_error)
        mase.append(np.mean(np.absolute(scaled_error)))
    mase = np.array(mase)
    if print_errors:
        print('Scaled error (SE) = ', se)
        print('Mean absolute scaled error (MASE) = ', mase)

    return mase, smape

