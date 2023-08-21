# Multiplication-Predictor

## A simple Neural Network with PyTorch

This code provides a simple implementation of a feedforward neural network using PyTorch to predict multiplication ;). 

## Overview

1. Sets up device configuration.
2. Defines a simple three-layer neural network.
3. Generates a dataset based on a multiplication operation.
4. Splits the dataset into training and testing.
5. Converts the dataset to tensors.
6. Configures the loss function and optimizer.
7. Trains the neural network.
8. Tests the neural network.
9. Sample input inference.

## Requirements

- PyTorch
- torchvision

## Usage

Run the provided Python script. This will train the neural network on a generated dataset and test its performance.

## Note

There is a comment about checking tensordict (https://github.com/pytorch-labs/tensordict) in case the normal dictionary approach does not work. Consider looking into this if you face any issues.

```bash
$ python <filename>.py
