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
from forecastNet import forecastnet
from train import train
from evaluate import evaluate
from demoDataset import generate_data

#Use a fixed seed for repreducible results
np.random.seed(1)

# Generate the dataset
train_data, test_data, valid_data, period = generate_data(T=2750, period = 50)

# Model parameters
model_type = 'dense2' #'dense' or 'conv', 'dense2' or 'conv2'
in_seq_length = 2 * period
hidden_dim = 24
out_seq_length = period
learning_rate = 0.0001
n_epochs= 100

# Initialise model
fcstnet = forecastnet(in_seq_length=in_seq_length, out_seq_length=out_seq_length, hidden_dim=hidden_dim,
                 learning_rate=learning_rate, n_epochs=n_epochs, save_file='./forecastnet3.ckpt', model=model_type)

# Train the model
training_costs, validation_costs = train(fcstnet, train_data, valid_data)
# Plot the training curves
plt.figure()
plt.plot(training_costs)
plt.plot(validation_costs)

# Evaluate the model
mase, smape = evaluate(fcstnet, test_data, return_lists=False)
print('')
print('MASE:', mase)
print('SMAPE:', smape)

# Generate and plot forecasts for various samples from the test dataset
start_idx = 20
for start_idx in [0, 50, 100]:
    test_sample = test_data[:, start_idx:]

    # Models with a Gaussian Mixture Density Component output
    if model_type == 'dense' or model_type == 'conv':
        # Generate a set of n_samples forecasts (Monte Carlo Forecasts)
        n_samples = 10 # 100 is a better value, but takes longer to compute
        batch_size = test_sample.shape[0]
        y_pred = np.zeros((batch_size, fcstnet.out_seq_length, n_samples))
        mu = np.zeros((batch_size, fcstnet.out_seq_length, n_samples))
        sigma = np.zeros((batch_size, fcstnet.out_seq_length, n_samples))
        for i in range(n_samples):
            print('Forecast sample', i)
            y_pred[:, :, i], mu[:, :, i], sigma[:, :, i] = fcstnet.forecast(test_sample)

        # Compute the Monte Carlo estimates of the mean and standard deviation
        s_mean = np.mean(y_pred, axis=2)
        s_std = np.std(y_pred, axis=2)
        botVarLine = s_mean - s_std
        topVarLine = s_mean + s_std

        # Plot the Monte Carlo mean and standard deviation
        plt.figure()
        plt.plot(np.arange(0, fcstnet.in_seq_length + fcstnet.out_seq_length),
                 test_sample[0, 0:fcstnet.in_seq_length + fcstnet.out_seq_length],
                 'o-', label='test_data')
        plt.plot(np.arange(fcstnet.in_seq_length, fcstnet.in_seq_length + fcstnet.out_seq_length),
                 s_mean[0, :],
                 '*-', linewidth=0.7, label='mean')
        plt.fill_between(np.arange(fcstnet.in_seq_length, fcstnet.in_seq_length + fcstnet.out_seq_length),
                         botVarLine[0, :],
                         topVarLine[0, :],
                         color='gray', alpha=0.3, label='Uncertainty')

    # Models with a linear output
    elif model_type == 'dense2' or model_type == 'conv2':
        # Generate a forecast
        y_pred = fcstnet.forecast(test_sample)

        # Plot the forecast
        plt.figure()
        plt.plot(np.arange(0, fcstnet.in_seq_length + fcstnet.out_seq_length),
                 test_sample[0, 0:fcstnet.in_seq_length + fcstnet.out_seq_length],
                 'o-', label='test_data')
        plt.plot(np.arange(fcstnet.in_seq_length, fcstnet.in_seq_length + fcstnet.out_seq_length),
                 y_pred[0, :],
                 '*-', linewidth=0.7, label='mean')

plt.show()
