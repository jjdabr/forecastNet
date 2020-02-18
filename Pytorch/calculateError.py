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
    Calculate various errors on a prediction Yhat given the ground truth Y. Both Yhat and Y can be in the following
    forms:
    * One dimensional arrays
    * Two dimensional arrays with several sequences along the first dimension (dimension 0).
    * Three dimensional arrays with several sequences along first dimension (dimension 0) and with the third dimension
      (dimension 2) being of size 1.
    :param Yhat: Prediction
    :param Y: Ground truth
    :param print_errors: If true the errors are printed.
    :return mase: Mean Absolute Scaled Error
    :return se: Scaled Error
    :return smape: Symmetric Mean Absolute Percentage Error
    :return nrmse: Normalised Root Mean Squared Error
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
    elif np.ndim(Y) == 3 and Y.shape[2] == 1:
        assert Y.shape[2] == 1, 'For a three dimensional array, Y.shape[2] == 1'
        Y = np.squeeze(Y, axis=2)
        assert Yhat.shape[2] == 1, 'For a three dimensional array, Y.shape[2] == 1'
        Yhat = np.squeeze(Yhat, axis=2)
        n_sequences = Y.shape[1]
    elif np.ndim(Y) == 3 and Y.shape[2] > 1:
        return calculate_miltidim_error(Yhat=Yhat, Y=Y, print_errors=print_errors)
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

    # Normalised Root Mean Squared Error
    nrmse = []
    for i in range(n_sequences):
        # Compute numerator and denominator
        numerator = 100 * np.sqrt(np.mean(np.square(Y[:, i] - Yhat[:, i])))
        denominator = np.max(Y[:, i]) - np.min(Y[:, i])
        # Remove any elements with zeros in the denominator
        non_zeros = denominator != 0
        numerator = numerator[non_zeros]
        denominator = denominator[non_zeros]
        # Calculate error
        nrmse.append(numerator / denominator)
    nrmse = np.array(nrmse)
    if print_errors:
        print('Normalised root mean squared error (NRMSE) = ', nrmse)

    return mase, se, smape, nrmse

def calculate_miltidim_error(Yhat, Y, print_errors=False):
    """
    Calculate various errors on a prediction Yhat given the ground truth Y. Both Yhat and Y are 3-dimensional
    arrays (tensors)
    :param Yhat: Prediction
    :param Y: Ground truth
    :return mase: Mean Absolute Scaled Error
    :return se: Scaled Error
    :return smape: Symmetric Mean Absolute Percentage Error
    :return nrmse: Normalised Root Mean Squared Error
    """

    # Prepare Y and Yhat based on their number of dimensions
    assert np.ndim(Y) == 3, 'Input must be three dimensional [n_sequences, n_batches, n_inputs]'

    [n_sequences, n_batches, n_inputs] = Y.shape
    # Symmetric Mean Absolute Percentage Error (M4 comp)

    smape = []
    for j in range(n_inputs):
        smape_seq = []
        for i in range(n_batches):
            # Compute numerator and denominator
            numerator = np.absolute(Y[:, i, j] - Yhat[:, i, j])
            denominator = (np.absolute(Y[:, i, j]) + np.absolute(Yhat[:, i, j]))
            # Remove any elements with zeros in the denominator
            non_zeros = denominator != 0
            numerator = numerator[non_zeros]
            denominator = denominator[non_zeros]
            # Sequence length
            length = numerator.shape[0]
            # Calculate error
            smape_seq.append(200.0 / length * np.sum(numerator / denominator))
        smape_seq = np.mean(np.array(smape_seq))
        smape.append(smape_seq)

    if print_errors:
        print('Symmetric mean absolute percentage error (sMAPE) = ', smape)

    # Mean absolute scaled error
    se = []
    mase = []
    for j in range(n_inputs):
        mase_seq = []
        for i in range(n_batches):
            numerator = (Y[:, i, j] - Yhat[:, i, j])
            denominator = np.sum(np.absolute(Y[1:, i, j] - Y[0:-1, i, j]), axis=0)
            # Check if denominator is zero
            if denominator == 0:
                warnings.warn("The denominator for the MASE is zero")
                se.append(np.NaN * np.ones(length))
                mase_seq.append(np.NaN)
                continue
            # Sequence length
            length = numerator.shape[0]
            # Scaled Error
            scaled_error = (length - 1) * numerator / denominator
            se.append(scaled_error)
            mase_seq.append(np.mean(np.absolute(scaled_error)))
        mase_seq = np.mean(np.array(mase_seq))
        mase.append(mase_seq)
    if print_errors:
        print('Scaled error (SE) = ', se)
        print('Mean absolute scaled error (MASE) = ', mase)

    # Normalised Root Mean Squared Error
    nrmse = []
    for j in range(n_inputs):
        nrmse_seq = []
        for i in range(n_batches):
            # Compute numerator and denominator
            numerator = 100 * np.sqrt(np.mean(np.square(Y[:, i, j] - Yhat[:, i, j])))
            denominator = np.max(Y[:, i, j]) - np.min(Y[:, i, j])
            # Remove any elements with zeros in the denominator
            non_zeros = denominator != 0
            numerator = numerator[non_zeros]
            denominator = denominator[non_zeros]
            # Calculate error
            nrmse_seq.append(numerator / denominator)
        nrmse_seq = np.mean(np.array(nrmse_seq))
        nrmse.append(nrmse_seq)
    if print_errors:
        print('Normalised root mean squared error (NRMSE) = ', nrmse)

    return mase, se, smape, nrmse


if __name__ == '__main__':
    x = np.reshape(np.arange(0, 10 * 2), (10, 2)) + np.random.rand(10,2)
    y = x + np.random.rand(10,2)

    # x = np.reshape(np.ones((10, 2)), (10, 2))
    # y = np.copy(x)
    # y[:,1] = y[:,1] + np.random.rand(10)

    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=2)

    mase, se, smape = calculate_error(x, y, print_errors=True)
