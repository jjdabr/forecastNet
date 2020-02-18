"""
ForecastNet with cells comprising densely connected layers.
forecastnet_graph provides the mixture density network outputs.
forecastnet_graph2 provides the linear outputs.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import tensorflow as tf
from gaussian import gaussian_sample, log_likelihood


def forecastnet_graph(X, Y, hidden_dim, out_seq_length, is_training):
    """
    Create a TensorFlow graph of ForecastNet with two densely connected layers in each cell.
    Each output is Gaussian mixture density network.
    :param X: A placeholder for the input to ForecastNet
    :param Y: A placeholder for the target of ForecastNet
    :param hidden_dim: Number of hidden units in each layer in the hidden cells.
    :param out_seq_length: Length of the output sequence of ForecastNet (number of steps-ahead to forecast)
    :param is_training: Indicator variable indicating if ForecastNet is being trained or tested.
    :return: outputs: The outputs holding the forecast. Size: [num_batches, out_seq_length]
    :return: outputs_mu: Mean of each Gaussian output. Size: [num_batches, out_seq_length]
    :return: outputs_sigma: Standard deviation of each Gaussian output. Size: [num_batches, out_seq_length]
    :return: cost: The MSE cost for a training sample.
    """
    # Create a empty list for the outputs
    outputs_mu = []
    outputs_sigma = []
    outputs = []
    # First hidden layer in the cell
    hidden = tf.layers.dense(inputs=X, units=hidden_dim, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             bias_initializer=tf.zeros_initializer(), name='hiddenA0')
    # Second hidden layer in the cell
    hidden = tf.layers.dense(inputs=hidden, units=hidden_dim, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             bias_initializer=tf.zeros_initializer(), name='hiddenB0')
    # First interleaved output
    output_mu = tf.layers.dense(inputs=hidden, units=1, activation=None, name='output_mu0')
    output_sigma = tf.layers.dense(inputs=hidden, units=1, activation=tf.nn.softplus, name='output_sigma0')
    output = tf.cond(is_training,
                     lambda: output_mu,
                     lambda: gaussian_sample(output_mu, output_sigma),
                     name='output0')
    # Add the output to the outputs list
    outputs_mu.append(output_mu)
    outputs_sigma.append(output_sigma)
    outputs.append(output)
    # Repeat for all outputs
    for i in range(1, out_seq_length):
        # Concatenate the input, the previous cell output and the previous output for the next cell input
        # if training, use the target as the next input, else use predicted output.
        concat = tf.cond(is_training,
                         lambda: tf.concat((X, hidden, tf.slice(Y, [0,i-1], [-1,1])), axis=1, name='concat' + str(i)),
                         lambda: tf.concat((X, hidden, output), axis=1, name='concat' + str(i)))
        # # Use the predicted output as the next input
        # concat = tf.concat((X, hidden, output), axis=1, name='concat' + str(i))

        # Next cell, first hidden layer
        hidden = tf.layers.dense(inputs=concat, units=hidden_dim, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 bias_initializer=tf.zeros_initializer(), name='hiddenA' + str(i))
        # Next cell, second hidden layer
        hidden = tf.layers.dense(inputs=hidden, units=hidden_dim, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 bias_initializer=tf.zeros_initializer(), name='hiddenB' + str(i))
        # Next interleaved output
        output_mu = tf.layers.dense(inputs=hidden, units=1, activation=None, name='output_mu' + str(i))
        output_sigma = tf.layers.dense(inputs=hidden, units=1, activation=tf.nn.softplus, name='output_sigma' + str(i))
        output = tf.cond(is_training,
                         lambda: output_mu,
                         lambda: gaussian_sample(output_mu, output_sigma),
                         name='output' + str(i))
        # Add next output to the outputs list
        outputs_mu.append(output_mu)
        outputs_sigma.append(output_sigma)
        outputs.append(output)
    # Convert the outputs list to a tensor
    outputs_mu = tf.concat(outputs_mu, axis=1, name='outputs_mu')
    outputs_sigma = tf.concat(outputs_sigma, axis=1, name='outputs_sigma')
    outputs = tf.concat(outputs, axis=1, name='outputs')

    # Cost using the log-likelihood of the Gaussian distribution
    cost = log_likelihood(Y, outputs_mu, outputs_sigma)

    return outputs, outputs_mu, outputs_sigma, cost


def forecastnet_graph2(X, Y, hidden_dim, out_seq_length, is_training):
    """
    Create a TensorFlow graph of ForecastNet with two densely connected layers in each cell.
    Each output is linear neuron.
    :param X: A placeholder for the input to ForecastNet
    :param Y: A placeholder for the target of ForecastNet
    :param hidden_dim: Number of hidden units in each layer in the hidden cells.
    :param out_seq_length: Length of the output sequence of ForecastNet (number of steps-ahead to forecast)
    :param is_training: Indicator variable indicating if ForecastNet is being trained or tested.
    :return: outputs: The outputs holding the forecast. Size: [num_batches, out_seq_length]
    :return: cost: The MSE cost for a training sample.
    """

    # Create a empty list for the outputs
    outputs = []
    # First hidden layer in the cell
    hidden = tf.layers.dense(inputs=X, units=hidden_dim, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             bias_initializer=tf.zeros_initializer(), name='hidden0')
    # Second hidden layer in the cell
    hidden = tf.layers.dense(inputs=hidden, units=hidden_dim, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             bias_initializer=tf.zeros_initializer(), name='hiddenB0')
    # First interleaved output
    output = tf.layers.dense(inputs=hidden, units=1, activation=None, name='tap0')
    # Add the output to the outputs list
    outputs.append(output)
    # Repeat for all outputs
    for i in range(1, out_seq_length):
        # Concatenate the input, the previous cell output and the previous output for the next cell input
        # if training, use the target as the next input, else use predicted output.
        concat = tf.cond(is_training,
                         lambda: tf.concat((X, hidden, tf.slice(Y, [0,i-1], [-1,1])), axis=1, name='concat' + str(i)),
                         lambda: tf.concat((X, hidden, output), axis=1, name='concat' + str(i)))
        # # Use the predicted output as the next input
        # concat = tf.concat((X, hidden, output), axis=1, name='concat' + str(i))
        # Next cell, first hidden layer
        hidden = tf.layers.dense(inputs=concat, units=hidden_dim, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 bias_initializer=tf.zeros_initializer(), name='hidden' + str(i))
        # Next cell, second hidden layer
        hidden = tf.layers.dense(inputs=hidden, units=hidden_dim, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 bias_initializer=tf.zeros_initializer(), name='hiddenB' + str(i))
        # Next interleaved output
        output = tf.layers.dense(inputs=hidden, units=1, activation=None, name='output' + str(i))
        # Add next output to the outputs list
        outputs.append(output)
    # Convert the outputs list to a tensor
    outputs = tf.concat(outputs, axis=1, name='outputs')

    # Cost using mean squared error (MSE)
    cost = tf.reduce_mean(tf.square(outputs - Y))

    return outputs, cost
