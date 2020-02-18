"""
Helper functions relating to the Gaussian Mixture Density Network.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import torch

def gaussian_loss(z, mu, sigma):
    """
    Calculate the negative log-likelihood of a given sample Y for a Gaussian with parameters mu and sigma.
    Note that this equation is specific for a single one-dimensional Gaussian mixture component.
    :param z: Target sample
    :param mu: Mean of the Gaussian
    :param sigma: Standard deviation of the Gaussian
    :return log_lik: The computed (negative) log-likelihood.
    """
    # n is the number of samples in the sequence * batch size
    # n = tf.to_float(tf.multiply(self.T_decoder, tf.shape(self.decoder_inputs)[1]))
    # n = tf.to_float(tf.multiply(tf.shape(z)[0], tf.shape(z)[1]))
    n = 1.0
    # Calculate the NEGATIVE log likelihood (negative because we are doing gradient descent)
    loglik = torch.mean(n * torch.log(sigma) + 0.5 * ((z - mu) ** 2 / sigma ** 2))
    # loglik = tf.reduce_mean(0.5 * self.T_in * tf.log(tf.square(sigma)) + 0.5 * tf.div(tf.square(z - mu), tf.square(sigma)))
    return loglik

def mse_loss(input, target):
    """
    Calculate the mean squared error loss
    :param input: Input sample
    :param target: Target sample
    :return mse: The mean squared error
    """
    mse = torch.mean((input - target) ** 2)
    return mse