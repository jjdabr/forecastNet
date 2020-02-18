# ForecastNet

TensorFlow implementation of ForecastNet described in the paper entitled 
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting" 
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman.

Link to the paper: [https://arxiv.org/abs/2002.04155](https://arxiv.org/abs/2002.04155)

ForecastNet is a deep feed-forward neural network multi-step-ahead forecasting of time-series data. The model is designed for (but is not limited to) seasonal time-series data. It comprises a set of outputs which are interleaved between a series of "cells" (a term borrowed from RNN literature). Each cell is a feed-forward neural network which can be chosen according to your needs. This code presents ForecastNet with two different cell architectures: one comprising densely connected layers, and one comprising a convolutional neural network (CNN).

The key benifits of ForecastNet are:
1. It is a time-variant model, as opposed to a time-invariant model (In the paper we show that RNN and CNN models are time-invariant).
2. It naturally increases in complexity with increasing forecast reach.
3. It's interleaved outputs assist with convergence and mitigating vanishing-gradient problems.
4. The "cell" architecture is highly flexible.
5. It is shown to out-perform state of the art deep learning models and statistical models.

## Files

- demo.py: Trains and evaluates ForecastNet on a synthetic dataset.
- forecastNet.py: Contains the main class for ForecastNet.
- denseForecastNet.py: Contains functions to build the TensorFlow graph for ForecastNet with densely connected hidden cells.
- convForecastNet.py: Contains functions to build the TensorFlow graph for ForecastNet with convolutional hidden cells.
- train.py: Contains a rudimentary training function to train ForecastNet.
- evaluate.py: Contains a rudimentary training function to train ForecastNet.
- dataHelpers.py: Functions to generate the dataset use in demo.py and for for formatting data.
- gaussian.py: Contains helper functions for the Gaussian mixture density network output layer.
- calculateError.py: Contains helper functions to compute error metrics

## Usage

Run the demo.py script to train and evaluate ForecastNet model on a synthetic dataset. You can write your own graph structures by modifying denseForecastNet.py or convForecastNet.py.

## Notes
 
- The training function in train.py could be improved by using PyTorch a dataloader.

## Requirements

- Python 3.6
- Torch version 1.2.0
- NumPy 1.14.6.

