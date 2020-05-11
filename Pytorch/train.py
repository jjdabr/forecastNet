"""
A training function for ForecastNet. This code could be improved by using a PyTorch dataloader

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import torch
import time
from dataHelpers import format_input
from gaussian import gaussian_loss, mse_loss
import torch.nn.functional as F

# Set plot_train_progress to True if you want a plot of the forecast after each epoch
plot_train_progress = False
if plot_train_progress:
    import matplotlib.pyplot as plt

def train(fcstnet, train_x, train_y, validation_x=None, validation_y=None, restore_session=False):
    """
    Train the ForecastNet model on a provided dataset. 
    In the following variable descriptions, the input_seq_length is the length of the input sequence 
    (2*seasonal_period in the paper) and output_seq_length is the number of steps-ahead to forecast 
    (seasonal_period in the paper). The n_batches is the total number batches in the dataset. The 
    input_dim and output_dim are the dimensions of the input sequence and output sequence respectively
    (in the paper univariate sequences were used where input_dim=output_dim=1).
    :param fcstnet: A forecastNet object defined by the class in forecastNet.py. 
    :param train_x: Input training data in the form [input_seq_length, n_batches, input_dim]
    :param train_y: Target training data in the form [output_seq_length, n_batches, output_dim]
    :param validation_x: Optional input validation data in the form [input_seq_length, n_batches, input_dim]
    :param validation_y: Optional target validation data in the form [output_seq_length, n_batches, output_dim]
    :param restore_session: If true, restore parameters and keep training, else train from scratch
    :return: training_costs: a list of training costs over the set of epochs
    :return: validation_costs: a list of validation costs over the set of epochs
    """

    # Convert numpy arrays to Torch tensors
    if type(train_x) is np.ndarray:
        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    if type(train_y) is np.ndarray:
        train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
    if type(validation_x) is np.ndarray:
        validation_x = torch.from_numpy(validation_x).type(torch.FloatTensor)
    if type(validation_y) is np.ndarray:
        validation_y = torch.from_numpy(validation_y).type(torch.FloatTensor)

    # Format inputs
    train_x = format_input(train_x)
    validation_x = format_input(validation_x)

    validation_x = validation_x.to(fcstnet.device)
    validation_y = validation_y.to(fcstnet.device)

    # Initialise model with predefined parameters
    if restore_session:
        # Load model parameters
        checkpoint = torch.load(fcstnet.save_file)
        fcstnet.model.load_state_dict(checkpoint['model_state_dict'])
        fcstnet.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Number of batch samples
    n_samples = train_x.shape[0]

    # List to hold the training costs over each epoch
    training_costs = []
    validation_costs = []

    # Set in training mode
    fcstnet.model.train()

    # Training loop
    for epoch in range(fcstnet.n_epochs):

        # Start the epoch timer
        t_start = time.time()

        # Print the epoch number
        print('Epoch: %i of %i' % (epoch + 1, fcstnet.n_epochs))

        # Initial average epoch cost over the sequence
        batch_cost = []

        # Counter for permutation loop
        count = 0

        # Permutation to randomly sample from the dataset
        permutation = np.random.permutation(np.arange(0, n_samples, fcstnet.batch_size))

        # Loop over the permuted indexes, extract a sample at that index and run it through the model
        for sample in permutation:
            # Extract a sample at the current permuted index
            input = train_x[sample:sample + fcstnet.batch_size, :]
            target = train_y[:, sample:sample + fcstnet.batch_size, :]

            # Send input and output data to the GPU/CPU
            input = input.to(fcstnet.device)
            target = target.to(fcstnet.device)

            # Zero the gradients
            fcstnet.optimizer.zero_grad()
            # Compute outputs and loss for a mixture density network output
            if fcstnet.model_type == 'dense' or fcstnet.model_type == 'conv':
                # Calculate the outputs
                outputs, mu, sigma = fcstnet.model(input, target, is_training=True)
                # Calculate the loss
                loss = gaussian_loss(z=target, mu=mu, sigma=sigma)
            # Compute outputs and loss for a linear output
            elif fcstnet.model_type == 'dense2' or fcstnet.model_type == 'conv2':
                # Calculate the outputs
                outputs = fcstnet.model(input, target, is_training=True)
                loss = F.mse_loss(input=outputs, target=target)
            batch_cost.append(loss.item())
            # Calculate the derivatives
            loss.backward()
            # Update the model parameters
            fcstnet.optimizer.step()

            if count % 50 == 0:
                print("Average cost after training batch %i of %i: %f" % (count, permutation.shape[0], loss.item()))
            count += 1
        # Find average cost over sequences and batches
        epoch_cost = np.mean(batch_cost)
        # Calculate the average training cost over the sequence
        training_costs.append(epoch_cost)

        # Plot an animation of the training progress
        if plot_train_progress:
            plt.cla()
            plt.plot(np.arange(input.shape[0], input.shape[0] + target.shape[0]), target[:, 0, 0])
            temp = outputs.detach()
            plt.plot(np.arange(input.shape[0], input.shape[0] + target.shape[0]), temp[:, 0, 0])
            plt.pause(0.1)

        # Validation tests
        if validation_x is not None:
            fcstnet.model.eval()
            with torch.no_grad():
                # Compute outputs and loss for a mixture density network output
                if fcstnet.model_type == 'dense' or fcstnet.model_type == 'conv':
                    # Calculate the outputs
                    y_valid, mu_valid, sigma_valid = fcstnet.model(validation_x, validation_y, is_training=False)
                    # Calculate the loss
                    loss = gaussian_loss(z=validation_y, mu=mu_valid, sigma=sigma_valid)
                # Compute outputs and loss for a linear output
                elif fcstnet.model_type == 'dense2' or fcstnet.model_type == 'conv2':
                    # Calculate the outputs
                    y_valid = fcstnet.model(validation_x, validation_y, is_training=False)
                    # Calculate the loss
                    loss = F.mse_loss(input=y_valid, target=validation_y)
                validation_costs.append(loss.item())
            fcstnet.model.train()

        # Print progress
        print("Average epoch training cost: ", epoch_cost)
        if validation_x is not None:
            print('Average validation cost:     ', validation_costs[-1])
        print("Epoch time:                   %f seconds" % (time.time() - t_start))
        print("Estimated time to complete:   %.2f minutes, (%.2f seconds)" %
              ((fcstnet.n_epochs - epoch - 1) * (time.time() - t_start) / 60,
               (fcstnet.n_epochs - epoch - 1) * (time.time() - t_start)))

        # Save a model checkpoint
        best_result = False
        if validation_x is None:
            if training_costs[-1] == min(training_costs):
                best_result = True
        else:
            if validation_costs[-1] == min(validation_costs):
                best_result = True
        if best_result:
            torch.save({
                'model_state_dict': fcstnet.model.state_dict(),
                'optimizer_state_dict': fcstnet.optimizer.state_dict(),
            }, fcstnet.save_file)
            print("Model saved in path: %s" % fcstnet.save_file)

    return training_costs, validation_costs
