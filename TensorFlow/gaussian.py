"""
Helper functions relating to the Gaussian Mixture Density Network.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture
for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import tensorflow as tf


def gaussian_sample(mu, sigma):
    """
    Generate a sample from a Gaussian distribution given the mean (mu) and the standard deviation (sigma)
    :param mu: Mean of the Gaussian
    :param sigma: Standard deviation of the Gaussian
    :return sample: The generated sample.
    """
    # gauss_sample = np.random.normal()
    # gauss_sample = tf.random.normal(shape=tf.shape(mu))
    gauss_sample = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0)
    sample = tf.add(mu, tf.multiply(sigma, gauss_sample))
    return sample


def log_likelihood(Y, mu, sigma):
    """
    Calculate the negative log-likelihood of a given sample Y for a Gaussian with parameters mu and sigma.
    Note that this equation is specific for a single one-dimensional Gaussian mixture component.
    :param Y: Input sample
    :param mu: Mean of the Gaussian
    :param sigma: Standard deviation of the Gaussian
    :return log_lik: The computed (negative) log-likelihood.
    """
    log_lik = tf.reduce_mean(
        tf.log(sigma) + 0.5 * tf.div(tf.square(Y - mu), tf.square(sigma))
    )

    return log_lik
