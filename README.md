# ForecastNet

Implementation of ForecastNet described in the paper entitled
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman.

Link to the paper: [https://arxiv.org/abs/2002.04155](https://arxiv.org/abs/2002.04155)

ForecastNet is a deep feed-forward neural network multi-step-ahead forecasting of time-series data. The model is
designed for (but is not limited to) seasonal time-series data. It comprises a set of outputs which are interleaved
between a series of "cells" (a term borrowed from RNN literature). Each cell is a feed-forward neural network which can
be chosen according to your needs. This code presents ForecastNet with two different cell architectures: one comprising
densely connected layers, and one comprising a convolutional neural network (CNN).

The key benefits of ForecastNet are:
1. It is a time-variant model, as opposed to a time-invariant model (In the paper we show that RNN and CNN models are time-invariant).
2. It naturally increases in complexity with increasing forecast reach.
3. It's interleaved outputs assist with convergence and mitigating vanishing-gradient problems.
4. The "cell" architecture is highly flexible.
5. It is shown to out-perform state of the art deep learning models and statistical models.

## Usage Notes

- A PyTorch implementation and a TensorFlow implementation are provided. The PyTorch implementation is recommended as
it is is more complete and more generic.
- Both implementations provide a demonstration using a synthetic dataset. The PyTorch implementation will be easiest to
adapt to your own dataset.
- The TensorFlow implementation is written for univariate time-series. The PyTorch implementation accepts multivariate
datasets (it has been tested with univariate datasets and datasets with multivariate inputs and a univariate output).
- Please read the README files in each implementation's directory.
