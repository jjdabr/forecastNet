"""
This file contains the class which constructs the TensorFlow graph of ForecastNet
and provides a function for forecasting.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture
for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import tensorflow as tf

from denseForecastNet import forecastnet_graph, forecastnet_graph2
from convForecastNet import forecastnet_conv_graph, forecastnet_conv_graph2


class forecastnet:
    """
    Forecastnet class for implementing the TensorFlow graph
    and operations for the ForecastNet model.
    """

    def __init__(
        self,
        in_seq_length,
        out_seq_length,
        hidden_dim,
        n_epochs=1500,
        learning_rate=0.0001,
        save_file='./forecastnet.ckpt',
        model='dense',
    ):
        """
        Initialise forecastnet parameters and create the forecastnet TensorFlow graph.
        :param in_seq_length: Length of the input sequence to ForecastNet
        :param hidden_dim: Number of hidden units in each cell's hidden layer
        :param out_seq_length: Length of the output sequence of ForecastNet (number of steps-ahead to forecast)
        :param n_epochs: Number of epochs to train over
        :param learning_rate: Learning rate for the ADAM algorithm
        :param model: Use 'dense' for a two layered densely connected hidden cell and Mixture Density network outputs.
                      Use 'conv' for the convolutional hidden cell and Mixture Density network outputs.
                      Use 'dense2' for a two layered densely connected hidden cell and linear outputs.
                      Use 'conv2' for the convolutional hidden cell and linear outputs.
        """
        # Initialize variables passed
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.save_file = save_file
        self.model = model

        # Reset the default graph
        tf.reset_default_graph()

        # Set random seed to keep consistent results
        # tf.set_random_seed(1)

        # Create the placeholders for the TensorFlow graph
        self.X, self.Y, self.is_training = self.create_placeholders()

        # Build the TensorFlow graph
        self.build_graph()

        # Define the tensorflow optimizer. Use an AdamOptimizer.
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            self.cost
        )

        # Print the number of trainable parameters of the model
        print(
            'Trainable variables = ',
            np.sum([
                np.prod(v.get_shape().as_list())
                for v in tf.trainable_variables()
            ]),
        )
        print('')

    def create_placeholders(self):
        """
        Create the placeholders for the TensorFlow graph
        :return: X: the inputs to ForecastNet. Size: [n_batches x in_seq_length]
        :return: Y: the target outputs to forecastnet for training. Size: [n_batches x out_seq_length]
        :return: is_training: an indicator variable to indicate when ForecastNet is training or predicting.
        """
        # Create Placeholders of shape (n_x, n_y)
        X = tf.placeholder(tf.float32, shape=(None, self.in_seq_length), name="X")
        Y = tf.placeholder(tf.float32, shape=(None, self.out_seq_length), name="Y")
        # Indicator to indicate when training or testing
        is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        return X, Y, is_training

    def build_graph(self):
        """
        Build the TensorFlow graph for the specified model
        """
        if self.model == 'dense':
            # ForecastNet with two densely connected hidden layers in a cell
            # and Mixture Density Network outputs
            self.outputs, self.mu, self.sigma, self.cost = forecastnet_graph(
                self.X, self.Y, self.hidden_dim, self.out_seq_length, self.is_training
            )
        elif self.model == 'conv':
            # ForecastNet with a convlutional neural network in a cell
            # and Mixture Density Network outputs
            self.outputs, self.mu, self.sigma, self.cost = forecastnet_conv_graph(
                self.X, self.Y, self.hidden_dim, self.out_seq_length, self.is_training
            )
        elif self.model == 'dense2':
            # ForecastNet with two densely connected hidden layers in a cell
            # and linear outputs
            self.outputs, self.cost = forecastnet_graph2(
                self.X, self.Y, self.hidden_dim, self.out_seq_length, self.is_training
            )
        elif self.model == 'conv2':
            # ForecastNet with a convolutional neural network in a cell
            # and linear outputs
            self.outputs, self.cost = forecastnet_conv_graph2(
                self.X, self.Y, self.hidden_dim, self.out_seq_length, self.is_training
            )

    def forecast(self, test_data):
        """
        Perform a forecast with the input data: test_data.
        Note that only the first in_seq_length values in test_data
        are used are used to forecast the next out_seq_length values.
        It is best practice to compute several forecasts
        and average over the y_pred of all forecasts.
        :param test_data: the input sequence with shape [n_batches, in_seq_length]
        :return: y_pred: The outputs with the sampled forecast. Size: [num_batches, out_seq_length]
        :return: mu: Mean of each Gaussian output. Size: [num_batches, out_seq_length]
        :return: sigma: Standard deviation of each Gaussian output. Size: [num_batches, out_seq_length]
        """
        # Add saver to save and restore all the variables.
        saver = tf.train.Saver()

        # Input data dimensions
        n_batches = test_data.shape[0]
        # T = test_data.shape[1]

        # Limit the length of the input data
        test_data = test_data[:, : self.in_seq_length]

        # Compute the forecast
        with tf.Session() as sess:
            saver.restore(sess, self.save_file)
            if self.model == 'dense' or self.model == 'conv':
                y_pred, mu, sigma = sess.run(
                    (self.outputs, self.mu, self.sigma),
                    feed_dict={
                        self.X: test_data,
                        self.Y: np.empty((n_batches, self.out_seq_length)),
                        self.is_training: False,
                    },
                )
            elif self.model == 'dense2' or self.model == 'conv2':
                y_pred = sess.run(
                    self.outputs,
                    feed_dict={
                        self.X: test_data,
                        self.Y: np.empty((n_batches, self.out_seq_length)),
                        self.is_training: False,
                    },
                )
        if self.model == 'dense' or self.model == 'conv':
            return y_pred, mu, sigma
        elif self.model == 'dense2' or self.model == 'conv2':
            return y_pred
