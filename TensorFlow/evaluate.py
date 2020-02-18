"""
Code to Evaluate of using the Mean Absolute Scaled Error (MASE) and the Symmetric Mean Absolute Percentage Error (SMAPE)
of ForecastNet for a given test set.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import tensorflow as tf
from calculateError import calculate_error

def evaluate(model, test_data, return_lists=False):
    """
    Evaluate the model's forecasting ability on the test set. Samples are drawn from the test set using a sliding
    window and the Mean Absolute Scaled Error (MASE) and the Symmetric Mean Absolute Percentage Error (SMAPE) are
    computed for each sample.
    :param model: A forecastNet object defined by the class in forecastNet.py
    :param test_data: The test dataset with shape [n_batches, n_samples].
    :param return_lists: True if a list of MASE and SMAPE results for each sample are to
                         be provided. Otherwise, an average over the list is returned.
    :return: mase: Mean Absolute Scaled Error (MASE)
    :return: smape: Symmetric Mean Absolute Percentage Error (SMAPE)
    """
    # Add saver to save and restore all the variables.
    saver = tf.train.Saver()

    # Dataset dimensions
    n_batches = test_data.shape[0]
    T = test_data.shape[1]

    # Create the session to evaluate the model
    with tf.Session() as sess:
        # Restore previously saved parameters
        saver.restore(sess, model.save_file)
        # Create a list of indices over which the sampples are to be extracted from
        indices = np.arange(T - model.in_seq_length - model.out_seq_length)
        # Lists holding the MASE and SMAPE for each sample.
        mase_list = []
        smape_list = []
        # Loop over the indices, extract a sample at that index and evaluate the performace for the sample
        for t in indices:
            # Extract a sample at the current permuted index
            X_sample = test_data[:, t: t + model.in_seq_length]
            Y_sample = test_data[:, t + model.in_seq_length: t + model.in_seq_length + model.out_seq_length]
            # Mixture Density Network outputs require several Monte Carlo forecasts.
            # Linear outputs require a single forecast.
            if model.model == 'dense' or model.model == 'conv':
                n_forecasts = 100
            elif model.model == 'dense2' or model.model == 'conv2':
                n_forecasts = 1
            y_pred_list = []
            for i in range(n_forecasts):
                y_pred, cost = sess.run((model.outputs, model.cost),
                                            feed_dict={model.X: X_sample,
                                                       model.Y: Y_sample,
                                                       model.is_training: False})
                y_pred_list.append(y_pred)
            y_pred = np.mean(y_pred_list, axis=0)
            # Compute the MASE and SMAPE
            mase, smape = calculate_error(y_pred.T, Y_sample.T)
            mase_list.append(mase)
            smape_list.append(smape)
    # Average MASE and SMAPE over all the forecast samples.
    mase = np.mean(mase_list)
    smape = np.mean(smape_list)
    print('Average MASE =', mase)
    print('Average SMAPE =', smape)
    if return_lists:
        return np.ndarray.flatten(np.array(mase_list)), np.ndarray.flatten(np.array(smape_list))
    else:
        return mase, smape