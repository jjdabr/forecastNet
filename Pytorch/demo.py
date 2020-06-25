"""
This script demonstrates initialisation, training, evaluation, and forecasting of ForecastNet. The dataset used for the
time-invariance test in section 6.1 of the ForecastNet paper is used for this demonstration.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import matplotlib.pyplot as plt
from forecastNet import forecastNet
from train import train
from evaluate import evaluate
from dataHelpers import generate_data

# Use a fixed seed for repreducible results
np.random.seed(1)

# Generate the dataset
train_x, train_y, test_x, test_y, valid_x, valid_y, period = generate_data(
    T=2750, period=50, n_seqs=4
)
# train_data, test_data, valid_data, period = generate_data(T=1000, period = 10)

# Model parameters
model_type = 'dense2'  # 'dense' or 'conv', 'dense2' or 'conv2'
in_seq_length = 2 * period
out_seq_length = period
hidden_dim = 24
input_dim = 1
output_dim = 1
learning_rate = 0.0001
n_epochs = 100
batch_size = 16

# Initialise model
fcstnet = forecastNet(
    in_seq_length=in_seq_length,
    out_seq_length=out_seq_length,
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    model_type=model_type,
    batch_size=batch_size,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    save_file='./forecastnet.pt',
)

# Train the model
training_costs, validation_costs = train(
    fcstnet, train_x, train_y, valid_x, valid_y, restore_session=False
)
# Plot the training curves
plt.figure()
plt.plot(training_costs)
plt.plot(validation_costs)

# Evaluate the model
mase, smape, nrmse = evaluate(fcstnet, test_x, test_y, return_lists=False)
print('')
print('MASE:', mase)
print('SMAPE:', smape)
print('NRMSE:', nrmse)

# Generate and plot forecasts for various samples from the test dataset
samples = [0, 500, 1039]
# Models with a Gaussian Mixture Density Component output
if model_type == 'dense' or model_type == 'conv':
    # Generate a set of n_samples forecasts (Monte Carlo Forecasts)
    num_forecasts = 10
    y_pred = np.zeros((test_y.shape[0], len(samples), test_y.shape[2], num_forecasts))
    mu = np.zeros((test_y.shape[0], len(samples), test_y.shape[2], num_forecasts))
    sigma = np.zeros((test_y.shape[0], len(samples), test_y.shape[2], num_forecasts))
    for i in range(num_forecasts):
        y_pred[:, :, :, i], mu[:, :, :, i], sigma[:, :, :, i] = fcstnet.forecast(
            test_x[:, samples, :]
        )
    s_mean = np.mean(y_pred, axis=3)
    s_std = np.std(y_pred, axis=3)
    botVarLine = s_mean - s_std
    topVarLine = s_mean + s_std

    for i in range(len(samples)):
        plt.figure()
        plt.plot(
            np.arange(0, in_seq_length),
            test_x[:, samples[i], 0],
            '-o',
            label='input',
        )
        plt.plot(
            np.arange(in_seq_length, in_seq_length + out_seq_length),
            test_y[:, samples[i], 0],
            '-o',
            label='data',
        )
        plt.plot(
            np.arange(in_seq_length, in_seq_length + out_seq_length),
            s_mean[:, i, 0],
            '-*',
            label='forecast',
        )
        plt.fill_between(
            np.arange(in_seq_length, in_seq_length + out_seq_length),
            botVarLine[:, i, 0],
            topVarLine[:, i, 0],
            color='gray',
            alpha=0.3,
            label='Uncertainty',
        )
        plt.legend()
# Models with a linear output
elif model_type == 'dense2' or model_type == 'conv2':
    # Generate a forecast
    y_pred = fcstnet.forecast(test_x[:, samples, :])

    for i in range(len(samples)):
        # Plot the forecast
        plt.figure()
        plt.plot(
            np.arange(0, fcstnet.in_seq_length),
            test_x[:, samples[i], 0],
            'o-',
            label='test_data',
        )
        plt.plot(
            np.arange(
                fcstnet.in_seq_length, fcstnet.in_seq_length + fcstnet.out_seq_length
            ),
            test_y[:, samples[i], 0],
            'o-',
        )
        plt.plot(
            np.arange(
                fcstnet.in_seq_length, fcstnet.in_seq_length + fcstnet.out_seq_length
            ),
            y_pred[:, i, 0],
            '*-',
            linewidth=0.7,
            label='mean',
        )

plt.show()
