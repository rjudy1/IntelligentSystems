# author: Rachael Judy
# date: 7 November 2023
# purpose: develop a multilayer neural network with backpropagation to differentiate the MNIST digits as well as
#           an autoencoder with the same number of hidden neurons as the MNIST digits


from collections import defaultdict
import csv
import datetime
import random
import time

from neuron import Neuron, ActivationFunction


class Network:
    def __init__(self, neurons_by_layer: list, activations_by_layer: list = None,
                 weight_files: [str] = None, momentum_alpha: float = 0.0, learning_rate=.0005,
                 special_activations: list = None):
        """
        make a list of lists of neurons

        expect weight_files, if not None, to have a list of files, one per layer

        if using special activations, must be of size matching number of layers
        """

        if special_activations is None:
            self._special_activations = [None for i in range(len(neurons_by_layer) - 1)]
        else:
            self._special_activations = special_activations
        # default to reading from weight file to initialize neurons
        if weight_files is not None:
            weights = [[] for _ in range(len(neurons_by_layer) - 1)]
            for idx, file in enumerate(weight_files):
                if file is None:
                    weights[idx] = [[(random.random() - .5) / 32 for _ in range(neurons_by_layer[idx]+1)]
                                    for _ in range(neurons_by_layer[idx+1])] # must generate the bias initialization
                else:
                    weights[idx] = self.read_weights(file)

            if activations_by_layer is None:
                activations_by_layer = [ActivationFunction.SIGMOID for _ in weights]
            self._neurons = [[Neuron(len(neuron) - 1, neuron[1:], neuron[0], learning_rate=learning_rate,
                                     momentum_alpha=momentum_alpha, activation_function=activation)
                              for neuron in weight_layer]
                             for idx, (activation, weight_layer) in enumerate(zip(activations_by_layer, weights))]
        elif neurons_by_layer is not None:
            if activations_by_layer is None:
                activations_by_layer = [ActivationFunction.SIGMOID for _ in neurons_by_layer[1:]]

            # create neurons with default weights
            self._neurons = [[Neuron(neurons_by_layer[i], learning_rate=learning_rate, momentum_alpha=momentum_alpha,
                                     activation_function=activations_by_layer[i])
                              for _ in range(neurons_by_layer[i + 1])] for i in range(len(neurons_by_layer) - 1)]
        else:
            raise Exception("Network details not specified by layers or weight file")

    def write_weights(self, filename: str) -> None:
        """
        write neuron weights as comma separated values, each line new neuron, empty line indicates line
        """
        for idx, layer in enumerate(self._neurons):
            with open(''.join([filename, '_layer', str(idx), '.csv']), 'w') as weight_file:
                for neuron in layer:
                    weight_file.write(
                        str(neuron.get_bias()) + ',' + ','.join([str(w) for w in neuron.get_weights()]) + '\n')
                # weight_file.write('\n')

    def read_weights(self, filename: str) -> []:
        with open(filename) as file:
            reader = csv.reader(file)
            layer_weights = [[float(weight) for weight in row] for row in reader]
        return layer_weights

    def propagate_inputs(self, inputs):
        """
        propogate inputs and return final layer neurons
        :param inputs:
        :return:
        """
        input_list = [inputs]
        for idx, layer in enumerate(self._neurons):
            if self._special_activations[idx] is not None:
                input_list.append(self._special_activations[idx](inputs, layer))
            else:
                input_list.append([neuron.activate(input_list[-1]) for neuron in layer])
        return input_list

    def determine_error(self, label, result: list, confusion_matrix: defaultdict = None):
        highest_probability_label = -1
        highest_probability = 0
        for idx, output in enumerate(result[-1]):
            if output > highest_probability:
                highest_probability = output
                highest_probability_label = idx
        if confusion_matrix is not None:
            confusion_matrix[label][highest_probability_label] += 1
        return 1 if highest_probability_label != label else 0

    def determine_immediate_error(self, label, idx, output, data):
        return (1 if idx == label else 0) - output

    def continue_training(self, current_error, desired_error, epoch, epoch_limit):
        if epoch_limit > 0 and desired_error > 0:
            if epoch >= epoch_limit:
                return False
            else:
                return current_error > desired_error
        elif epoch_limit > 0:
            return epoch < epoch_limit
        elif desired_error > 0:
            return current_error > desired_error
        else:
            return False

    def test(self, test_file: str) -> (float, float, float, float):
        errors, test_count = 0, 0
        confusion_matrix = defaultdict(lambda: defaultdict(lambda: 0))
        with open(test_file) as tests:
            for test in tests:
                label, image = int(test.split(' ')[0]), [float(i) for i in test.strip().split(' ')[1:]]
                result = self.propagate_inputs(image)
                errors += self.determine_error(label, result, confusion_matrix)
                test_count += 1
        return errors / test_count, confusion_matrix  # error fraction

    def train(self, training_file: str, epoch_size: int = 1000, desired_error_rate: float = .01, epoch_limit=-1,
              last_layer_only=False):
        error_fractions = []
        error_fraction = desired_error_rate * 1000
        epoch = 0
        with open(training_file) as file:
            trainings = file.readlines()
        start = time.time()
        while self.continue_training(error_fraction, desired_error_rate, epoch, epoch_limit):
            errors, total = 0, 0

            random.shuffle(trainings)
            for test in trainings[:epoch_size]:
                label, image = int(test.split(' ')[0]), [float(i) for i in test.strip().split(' ')[1:]]

                # test image in neuron, adjust weights
                result = self.propagate_inputs(image)
                highest_probability_label = -1
                highest_probability = 0
                for idx, output in enumerate(result[-1]):
                    if output > highest_probability:
                        highest_probability = output
                        highest_probability_label = idx
                errors += 1 if highest_probability_label != label else 0

                total += 1

                # compute all the errors
                backprop_errors = [[(1 if idx == label else 0) - output
                                    * self._neurons[-1][idx].get_derivative_of_activation(result[-2])
                                    for (idx, output) in enumerate(result[-1])]]

                if not last_layer_only:
                    for i in range(len(result) - 3, -1, -1):  # iterates over layers
                        backprop_errors.insert(0, [self._neurons[i][idx].get_derivative_of_activation(result[i])
                                                   * sum(self._neurons[i + 1][j].get_weights()[idx] * backprop_errors[0][j]
                                                         for j in range(len(self._neurons[i + 1])))
                                                   for (idx, output) in enumerate(result[i + 1])])

                    # apply the errors to all the weights
                    for i in range(len(backprop_errors)):  # go layer by layer, updating weights
                        for j in range(len(self._neurons[i])):
                            self._neurons[i][j].update_weights(backprop_errors[i][j], result[i])
                else:
                    for j in range(len(self._neurons[-1])):
                        self._neurons[-1][j].update_weights(backprop_errors[-1][j], result[-2])

            error_fraction = errors / total
            epoch += 1
            if epoch % 10 == 0:
                print(epoch, error_fraction, time.time() - start, datetime.datetime.now().strftime("%H:%M:%S"))
                start = time.time()
                error_fractions.append(error_fraction)
            if epoch % 50 == 0:
                self.write_weights(f'weights/weights{int(time.time())}_{epoch}')

        return error_fractions
