"""
This is a common neuron used for various networks
author: Rachael Judy

"""

from enum import Enum
import math
import numpy as np
import random


class ActivationFunction(Enum):
    STEP = 0
    LINEAR = 1
    SIGMOID = 2
    RELU = 3
    LEAKY_RELU = 4
    TAN_H = 5


class Neuron:
    def __init__(self, input_size: int, initial_weights: list = None,
                 bias: float = None, learning_rate: float = .005,
                 momentum_alpha=0,
                 activation_function=ActivationFunction.SIGMOID):
        self._weights = np.array(initial_weights) if initial_weights \
            else np.array([(random.random() - .5) / 32 for _ in range(input_size)])
        self._previous_weight_change = np.array([0 for _ in range(input_size + 1)])

        self._activation_function = activation_function
        self._bias = random.random() - .75 if bias is None else bias
        self._learning_rate = learning_rate
        self._momentum_parameter = momentum_alpha
        self._activation_parameter = None

    def get_weights(self) -> np.array:
        return self._weights

    def get_bias(self) -> float:
        return self._bias

    def set_bias(self, bias):
        self._bias = bias

    def set_activation_function(self, af: ActivationFunction, parameter=0):
        self._activation_function = af
        self._activation_parameter = parameter

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, rate: float):
        self._learning_rate = rate

    def get_logit(self, inputs):
        return inputs @ self._weights + self._bias
        # return sum(i[0] * i[1] for i in zip(inputs, self._weights)) + self._bias

    def activate(self, inputs) -> float:
        net_input = self.get_logit(inputs)
        if self._activation_function == ActivationFunction.STEP:
            return 1 if net_input > 0 else 0
        elif self._activation_function == ActivationFunction.LINEAR:
            return net_input
        elif self._activation_function == ActivationFunction.SIGMOID:
            return 1 / (1 + math.e ** -net_input)
        elif self._activation_function == ActivationFunction.TAN_H:
            return (math.e ** net_input - math.e ** -net_input) / (math.e ** net_input + math.e ** -net_input)
        elif self._activation_function == ActivationFunction.RELU:
            return max(0, net_input)
        elif self._activation_function == ActivationFunction.LEAKY_RELU:
            return max(.1 * net_input, net_input)

    def get_derivative_of_activation(self, inputs) -> float:
        net_input = self.get_logit(inputs)
        if self._activation_function == ActivationFunction.STEP:
            return math.inf  # error
        elif self._activation_function == ActivationFunction.LINEAR:
            return 1
        elif self._activation_function == ActivationFunction.SIGMOID:
            return 1 / (1 + math.e ** -net_input) * (1 - 1 / (1 + math.e ** -net_input))
        elif self._activation_function == ActivationFunction.TAN_H:
            return 1 - ((math.e ** net_input - math.e ** -net_input) / (
                    math.e ** net_input + math.e ** -net_input)) ** 2
        elif self._activation_function == ActivationFunction.RELU:
            return 1 if net_input > 0 else 0
        elif self._activation_function == ActivationFunction.LEAKY_RELU:
            return 1 if net_input > 0 else .1

    def update_weights(self, error_delta, data):
        if len(data) != len(self._weights):
            raise Exception("Size of input to neuron does not match")
        for i in range(len(self._weights)):
            self._weights[i] += self._learning_rate * error_delta * data[i] \
                                + self._momentum_parameter * self._previous_weight_change[i]
            self._previous_weight_change[i] = self._learning_rate * error_delta * data[i] \
                                              + self._momentum_parameter * self._previous_weight_change[i]

        self._bias += self._learning_rate * error_delta \
                      + self._momentum_parameter * self._previous_weight_change[-1]
        self._previous_weight_change[-1] = self._learning_rate * error_delta \
                                           + self._momentum_parameter * self._previous_weight_change[-1]
