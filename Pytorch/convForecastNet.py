"""
ForecastNet with cells comprising a convolutional neural network.
ForecastNetConvModel provides the mixture density network outputs.
ForecastNetConvModel2 provides the linear outputs.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ForecastNetConvModel(nn.Module):
    """
    Class for the convolutional hidden cell version of the model
    """
    def __init__(self, input_dim, hidden_dim, output_dim, in_seq_length, out_seq_length, device):
        """
        Constructor
        :param input_dim: Dimension of the inputs
        :param hidden_dim: Number of hidden units
        :param output_dim: Dimension of the outputs
        :param in_seq_length: Length of the input sequence
        :param out_seq_length: Length of the output sequence
        :param device: The device on which compuations are perfomed.
        """
        super(ForecastNetConvModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.device = device

        self.conv_layer1 = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=5, padding=2) for i in range(out_seq_length)])
        self.conv_layer2 = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1) for i in range(out_seq_length)])
        flatten_layer = [nn.Linear(hidden_dim * (input_dim * in_seq_length), hidden_dim)]
        for i in range(out_seq_length - 1):
            flatten_layer.append(nn.Linear(hidden_dim * (input_dim * in_seq_length + hidden_dim + output_dim), hidden_dim))
        self.flatten_layer = nn.ModuleList(flatten_layer)
        self.mu_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])
        self.sigma_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])

        # # Convolutional Layers with Pooling
        # self.conv_layer1 = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=5, padding=2) for i in range(out_seq_length)])
        # self.pool_layer1 = nn.ModuleList([nn.AvgPool1d(kernel_size=2, padding=0) for i in range(out_seq_length)])
        # self.conv_layer2 = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1) for i in range(out_seq_length)])
        # self.pool_layer2 = nn.ModuleList([nn.AvgPool1d(kernel_size=2, padding=0) for i in range(out_seq_length) for i in range(out_seq_length)])
        # flatten_layer = [nn.Linear(hidden_dim//4 * (input_dim * in_seq_length), hidden_dim)]
        # for i in range(out_seq_length - 1):
        #     flatten_layer.append(nn.Linear(hidden_dim * ((input_dim * in_seq_length + hidden_dim + output_dim) // 4), hidden_dim))
        # self.flatten_layer = nn.ModuleList(flatten_layer)
        # self.mu_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])
        # self.sigma_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])

    def forward(self, input, target, is_training=False):
        """
        Forward propagation of the convolutional ForecastNet model
        :param input: Input data in the form [input_seq_length, batch_size, input_dim]
        :param target: Target data in the form [output_seq_length, batch_size, output_dim]
        :param is_training: If true, use target data for training, else use the previous output.
        :return: outputs: Sampled forecast outputs in the form [decoder_seq_length, batch_size, input_dim]
        :return: mu: Outputs of the mean layer [decoder_seq_length, batch_size, input_dim]
        :return: sigma: Outputs of the standard deviation layer [decoder_seq_length, batch_size, input_dim]
        """
        # Initialise outputs
        outputs = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(self.device)
        mu = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(self.device)
        sigma = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(self.device)
        # First input
        next_cell_input = input.unsqueeze(dim=1)
        # Propagate through network
        for i in range(self.out_seq_length):
            # Propagate through the cell
            hidden = F.relu(self.conv_layer1[i](next_cell_input))
            # hidden = self.pool_layer1[i](hidden)
            hidden = F.relu(self.conv_layer2[i](hidden))
            # hidden = self.pool_layer2[i](hidden)
            hidden = hidden.reshape((input.shape[0], -1))
            hidden = F.relu(self.flatten_layer[i](hidden))

            # Calculate output
            mu_ = self.mu_layer[i](hidden)
            sigma_ = F.softplus(self.sigma_layer[i](hidden))
            mu[i,:,:] = mu_
            sigma[i,:,:] = sigma_
            outputs[i,:,:] = torch.normal(mu_, sigma_).to(self.device)

            # Prepare the next input
            if is_training:
                next_cell_input = torch.cat((input, hidden, target[i, :, :]), dim=1).unsqueeze(dim=1)
            else:
                next_cell_input = torch.cat((input, hidden, outputs[i, :, :]), dim=1).unsqueeze(dim=1)
            # Concatenate next input and
        return outputs, mu, sigma


class ForecastNetConvModel2(nn.Module):
    """
    Class for the convolutional hidden cell version of the model
    """
    def __init__(self, input_dim, hidden_dim, output_dim, in_seq_length, out_seq_length, device):
        """
        Constructor
        :param input_dim: Dimension of the inputs
        :param hidden_dim: Number of hidden units
        :param output_dim: Dimension of the outputs
        :param in_seq_length: Length of the input sequence
        :param out_seq_length: Length of the output sequence
        :param device: The device on which compuations are perfomed.
        """
        super(ForecastNetConvModel2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.device = device

        self.conv_layer1 = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=5, padding=2) for i in range(out_seq_length)])
        self.conv_layer2 = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1) for i in range(out_seq_length)])
        flatten_layer = [nn.Linear(hidden_dim * (input_dim * in_seq_length), hidden_dim)]
        for i in range(out_seq_length - 1):
            flatten_layer.append(nn.Linear(hidden_dim * (input_dim * in_seq_length + hidden_dim + output_dim), hidden_dim))
        self.flatten_layer = nn.ModuleList(flatten_layer)
        self.output_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])

        # # Convolutional Layers with Pooling
        # self.conv_layer1 = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=5, padding=2) for i in range(out_seq_length)])
        # self.pool_layer1 = nn.ModuleList([nn.AvgPool1d(kernel_size=2, padding=0) for i in range(out_seq_length)])
        # self.conv_layer2 = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1) for i in range(out_seq_length)])
        # self.pool_layer2 = nn.ModuleList([nn.AvgPool1d(kernel_size=2, padding=0) for i in range(out_seq_length) for i in range(out_seq_length)])
        # flatten_layer = [nn.Linear(hidden_dim//4 * (input_dim * in_seq_length), hidden_dim)]
        # for i in range(out_seq_length - 1):
        #     flatten_layer.append(nn.Linear(hidden_dim * ((input_dim * in_seq_length + hidden_dim + output_dim) // 4), hidden_dim))
        # self.flatten_layer = nn.ModuleList(flatten_layer)
        # self.output_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])

    def forward(self, input, target, is_training=False):
        """
        Forward propagation of the convolutional ForecastNet model
        :param input: Input data in the form [input_seq_length, batch_size, input_dim]
        :param target: Target data in the form [output_seq_length, batch_size, output_dim]
        :param is_training: If true, use target data for training, else use the previous output.
        :return: outputs: Forecast outputs in the form [decoder_seq_length, batch_size, input_dim]
        """
        # Initialise outputs
        outputs = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(self.device)
        # First input
        next_cell_input = input.unsqueeze(dim=1)
        # Propagate through network
        for i in range(self.out_seq_length):
            # Propagate through the cell
            hidden = F.relu(self.conv_layer1[i](next_cell_input))
            # hidden = self.pool_layer1[i](hidden)
            hidden = F.relu(self.conv_layer2[i](hidden))
            # hidden = self.pool_layer2[i](hidden)
            hidden = hidden.reshape((input.shape[0], -1))
            hidden = F.relu(self.flatten_layer[i](hidden))

            # Calculate output
            output = self.output_layer[i](hidden)
            outputs[i,:,:] = output

            # Prepare the next input
            if is_training:
                next_cell_input = torch.cat((input, hidden, target[i, :, :]), dim=1).unsqueeze(dim=1)
            else:
                next_cell_input = torch.cat((input, hidden, outputs[i, :, :]), dim=1).unsqueeze(dim=1)
            # Concatenate next input and
        return outputs
