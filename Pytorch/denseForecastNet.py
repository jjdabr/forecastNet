"""
ForecastNet with cells comprising densely connected layers.
ForecastNetDenseModel provides the mixture density network outputs.
ForecastNetDenseModel2 provides the linear outputs.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ForecastNetDenseModel(nn.Module):
    """
    Class for the densely connected hidden cells version of the model
    """
    def __init__(self, input_dim, hidden_dim, output_dim, in_seq_length, out_seq_length, device):
        """
        Constructor
        :param input_dim: Dimension of the inputs
        :param hidden_dim: Number of hidden units
        :param output_dim: Dimension of the outputs
        :param in_seq_length: Length of the input sequence
        :param out_seq_length: Length of the output sequence
        """
        super(ForecastNetDenseModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.device = device
        # Input dimension of componed inputs and sequences
        input_dim_comb = input_dim * in_seq_length

        # Initialise layers
        hidden_layer1 = [nn.Linear(input_dim_comb, hidden_dim)]
        for i in range(out_seq_length - 1):
            hidden_layer1.append(nn.Linear(input_dim_comb + hidden_dim + output_dim, hidden_dim))
        self.hidden_layer1 = nn.ModuleList(hidden_layer1)
        self.hidden_layer2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(out_seq_length)])
        self.mu_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])
        self.sigma_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])

    def forward(self, input, target, is_training=False):
        """
        Forward propagation of the dense ForecastNet model
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
        next_cell_input = input
        for i in range(self.out_seq_length):
            # Propagate through cell
            out = F.relu(self.hidden_layer1[i](next_cell_input))
            out = F.relu(self.hidden_layer2[i](out))
            # Calculate the output
            mu_ = self.mu_layer[i](out)
            sigma_ = F.softplus(self.sigma_layer[i](out))
            mu[i,:,:] = mu_
            sigma[i,:,:] = sigma_
            outputs[i,:,:] = torch.normal(mu_, sigma_).to(self.device)
            # Prepare the next input
            if is_training:
                next_cell_input = torch.cat((input, out, target[i, :, :]), dim=1)
            else:
                next_cell_input = torch.cat((input, out, outputs[i, :, :]), dim=1)
        return outputs, mu, sigma


class ForecastNetDenseModel2(nn.Module):
    """
    Class for the densely connected hidden cells version of the model
    """
    def __init__(self, input_dim, hidden_dim, output_dim, in_seq_length, out_seq_length, device):
        """
        Constructor
        :param input_dim: Dimension of the inputs
        :param hidden_dim: Number of hidden units
        :param output_dim: Dimension of the outputs
        :param in_seq_length: Length of the input sequence
        :param out_seq_length: Length of the output sequence
        """
        super(ForecastNetDenseModel2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.device = device
        # Input dimension of componed inputs and sequences
        input_dim_comb = input_dim * in_seq_length

        # Initialise layers
        hidden_layer1 = [nn.Linear(input_dim_comb, hidden_dim)]
        for i in range(out_seq_length - 1):
            hidden_layer1.append(nn.Linear(input_dim_comb + hidden_dim + output_dim, hidden_dim))
        self.hidden_layer1 = nn.ModuleList(hidden_layer1)
        self.hidden_layer2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(out_seq_length)])
        self.output_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])

    def forward(self, input, target, is_training=False):
        """
        Forward propagation of the dense ForecastNet model
        :param input: Input data in the form [input_seq_length, batch_size, input_dim]
        :param target: Target data in the form [output_seq_length, batch_size, output_dim]
        :param is_training: If true, use target data for training, else use the previous output.
        :return: outputs: Forecast outputs in the form [decoder_seq_length, batch_size, input_dim]
        """
        # Initialise outputs
        outputs = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(self.device)
        # First input
        next_cell_input = input
        for i in range(self.out_seq_length):
            # Propagate through cell
            hidden = F.relu(self.hidden_layer1[i](next_cell_input))
            hidden = F.relu(self.hidden_layer2[i](hidden))
            # Calculate the output
            output = self.output_layer[i](hidden)
            outputs[i,:,:] = output
            # Prepare the next input
            if is_training:
                next_cell_input = torch.cat((input, hidden, target[i, :, :]), dim=1)
            else:
                next_cell_input = torch.cat((input, hidden, outputs[i, :, :]), dim=1)
        return outputs