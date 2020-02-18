"""
A fairly rudimentary training function for ForecastNet. This function is specific to a particular data format and should
be replaced by a training function that suits your needs. For example, a proper data loader should be constructed that
partitions the samples into batches. This code assumes that the data comprises several streams of values. Each stream
is treated as a batch and the sequence is split up into samples using a sliding window. Please refer to the PyTorch code
for a more generic function.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import numpy as np
import tensorflow as tf
import time

def train(model, dataset, validation_set=np.array([]), restore_session=False):
    """
    Train the forecastnet model. It is assumed that the input data comprises several sequences of particular univariate
    variable . Each sequence is treated as a batch and the sequence is split up into samples using a sliding window.
    :param model: A forecastNet object defined by the class in forecastNet.py
    :param dataset: The training dataset with shape [n_batches, n_samples].
    :param validation_set: An optional validation dataset with shape [n_batches, n_samples].
    :param restore_session: True if the training must start with a set of parameters from a previously trained model.
    :return: training_costs, validation_costs: a list of training costs and validation costs for each epoch
    """

    # Dataset sequence length (dimension 0 is the batch size
    T = dataset.shape[1]

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Create a saver variable to save the state
    saver = tf.train.Saver()

    # Lists to hold the training and validation costs over each epoch
    training_costs = []
    validation_costs = []

    # Create the session to train the model
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Restore previously saved parameters
        if restore_session:
            saver.restore(sess, model.save_file)

        # Main training loop
        for epoch in range(model.n_epochs):
            t_start = time.time()

            # Initial average epoch cost over the sequence
            batch_cost = []

            # Sample counter for permutation loop
            count = 0

            # Create a random permutation of all the dataset indexes to perform random sampling from the dataset
            permutation = np.random.permutation(T - model.in_seq_length - model.out_seq_length)
            # Create a sequence from 0 to T - model.in_seq_length - model.out_seq_length
            # permutation = np.arange(T - model.in_seq_length - model.out_seq_length)

            # Randomly sample from the dataset (without replacement)
            for t in permutation:
                # Extract a sample at the current permuted index
                minibatch_X = dataset[:, t: t + model.in_seq_length]
                minibatch_Y = dataset[:, t + model.in_seq_length: t + model.in_seq_length + model.out_seq_length]
                # Train on the extracted sample
                _, minibatch_cost = sess.run([model.optimizer, model.cost],
                                             feed_dict={model.X: minibatch_X,
                                                        model.Y: minibatch_Y,
                                                        model.is_training: True})

                # Update the batch_cost list
                batch_cost.append(minibatch_cost)

                # Increment the sample count.
                count += 1

            # Find average cost over sequence
            epoch_cost = np.mean(batch_cost)
            # Update the training costs list
            training_costs.append(epoch_cost)

            # Print the cost every epoch
            if epoch % 1 == 0:  # and print_cost == True:
                # Print the progress
                print('Epoch: %i of %i' % (epoch + 1, model.n_epochs))
                print("Average epoch cost =    ", epoch_cost)
                # Calculate the validation cost
                if validation_set.size != 0:
                    len_valid = validation_set.shape[1]
                    validation_list = []
                    for idx in range(len_valid - model.in_seq_length - model.out_seq_length):
                        validation_X = validation_set[:, idx : idx + model.in_seq_length]
                        validation_Y = validation_set[:, idx + model.in_seq_length: idx + model.in_seq_length + model.out_seq_length]
                        if model.model == 'dense' or model.model == 'conv':
                            n_forecasts = 1#0
                        elif model.model == 'dense2' or model.model == 'conv2':
                            n_forecasts = 1
                        v_cost_list = []
                        for i in range(n_forecasts):
                            v_outpus, v_cost = sess.run((model.outputs, model.cost),
                                                        feed_dict={model.X: validation_X,
                                                                   model.Y: validation_Y,
                                                                   model.is_training: False})
                            v_mse = np.mean(np.square(v_outpus - validation_Y))
                            v_cost_list.append(v_mse)
                        validation_cost = np.mean(v_cost_list)
                        validation_list.append(validation_cost)
                    validation_costs.append(np.mean(validation_list))
                    print('Average validation MSE =', np.mean(validation_costs))
                # Print the epoch time and the expected time left to complete the training.
                print("Epoch time = %f seconds" % (time.time() - t_start))
                print("Estimated time to complete = %.2f minutes, (%.2f seconds)" %
                      ((model.n_epochs - epoch - 1) * (time.time() - t_start) / 60,
                       (model.n_epochs - epoch - 1) * (time.time() - t_start)))
                t_start = time.time()
            # Save the current parameters. This can be done less often
            save_path = saver.save(sess, model.save_file)
            print("Model saved in path: %s" % save_path)

    return training_costs, validation_costs