"""
Code to Evaluate of using the Mean Absolute Scaled Error (MASE) and the Symmetric Mean Absolute Percentage Error (SMAPE)
of ForecastNet for a given test set.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import torch
from dataHelpers import format_input
from calculateError import calculate_error

def evaluate(fcstnet, test_x, test_y, return_lists=False):
    """
    Calculate various error metrics on a test dataset
    :param fcstnet: A forecastNet object defined by the class in forecastNet.py
    :param test_x: Input test data in the form [encoder_seq_length, n_batches, input_dim]
    :param test_y: target data in the form [encoder_seq_length, n_batches, input_dim]
    :return: mase: Mean absolute scaled error
    :return: smape: Symmetric absolute percentage error
    :return: nrmse: Normalised root mean squared error
    """
    fcstnet.model.eval()

    # Load model parameters
    checkpoint = torch.load(fcstnet.save_file, map_location=fcstnet.device)
    fcstnet.model.load_state_dict(checkpoint['model_state_dict'])
    fcstnet.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    with torch.no_grad():
        if type(test_x) is np.ndarray:
            test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        if type(test_y) is np.ndarray:
            test_y = torch.from_numpy(test_y).type(torch.FloatTensor)

        # Format the inputs
        test_x = format_input(test_x)

        # Send to CPU/GPU
        test_x = test_x.to(fcstnet.device)
        test_y = test_y.to(fcstnet.device)

        # Number of batch samples
        n_samples = test_x.shape[0]

        # Inference
        y_pred_list = []
        # Compute outputs for a mixture density network output
        if fcstnet.model_type == 'dense' or fcstnet.model_type == 'conv':
            n_forecasts = 20
            for i in range(n_forecasts):
                y_pred, mu, sigma = fcstnet.model(test_x, test_y, is_training=False)
                y_pred_list.append(y_pred)
            y_pred = torch.mean(torch.stack(y_pred_list), dim=0)
        # Compute outputs for a linear output
        elif fcstnet.model_type == 'dense2' or fcstnet.model_type == 'conv2':
            y_pred = fcstnet.model(test_x, test_y, is_training=False)

        mase_list = []
        smape_list = []
        nrmse_list = []
        for i in range(n_samples):
            mase, se, smape, nrmse = calculate_error(y_pred[:, i, :].cpu().numpy(), test_y[:, i, :].cpu().numpy())
            mase_list.append(mase)
            smape_list.append(smape)
            nrmse_list.append(nrmse)
        # writer.close()
        mase = np.mean(mase_list)
        smape = np.mean(smape_list)
        nrmse = np.mean(nrmse_list)

    if return_lists:
        return np.ndarray.flatten(np.array(mase_list)), np.ndarray.flatten(np.array(smape_list)), np.ndarray.flatten(
            np.array(nrmse_list))
    else:
        return mase, smape, nrmse
