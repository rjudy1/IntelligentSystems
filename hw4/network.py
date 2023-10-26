# author: Rachael Judy
# date: 7 November 2023
# purpose: develop a multilayer neural network with backpropagation to differentiate the MNIST digits as well as
#           an autoencoder with the same number of hidden neurons as the MNIST digits


from collections import defaultdict
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from neuron import Neuron, ActivationFunction


# adapted from my hw 3
def create_data(image_file, label_file, data_limit, values: set, test_quantity: int, prefix: str) -> None:
    """
    This creates two files, a test and a training set for the values to classify from an assumed 500 values
    per item

    :param image_file: filename for data to sample from
    :param label_file: filename for labels
    :param data_limit: int indicating max line to sample from, not inclusive
    :param values: values to collect from files
    :param test_quantity: number of values to select from each class for the test set
    :param prefix: prefix for output filenames
    """

    # select test_quantity indices to put in the test and challenge set
    test_indices = set()
    while len(test_indices) < test_quantity:
        test_indices.add(random.randint(0, 499))

    training_data = list()
    test_file = open(''.join([prefix, 'test']), 'w')
    with open(image_file) as images, open(label_file) as labels:
        for index, (image, label) in enumerate(zip(images, labels)):
            if index % 500 not in test_indices and index < data_limit and int(label) in values:
                training_data.append(' '.join([label.strip(), image.strip().replace('\t', ' '), '\n']))
            elif index % 500 in test_indices and index < data_limit and int(label) in values:
                test_file.write(' '.join([label.strip(), image.strip().replace('\t', ' '), '\n']))

    training_file = open(''.join([prefix, 'training']), 'w')
    random.shuffle(training_data)
    training_file.writelines(training_data)
    training_file.close()
    test_file.close()


def plot_heatmaps(data: [[]], captions: [], image_caption: str):
    # plot heatmaps of weights initial and final

    fig, axes = plt.subplots(nrows=len(data), ncols=len(data[0]))
    fig.title(image_caption)
    fig.tight_layout()

    plt.subplots_adjust(left=0.6,
                        bottom=0.1,
                        right=0.6,
                        top=0.5,
                        wspace=0.4,
                        hspace=0.4)
    for i in range(len(data)):
        for j in range(len(data[0])):
            heatmap = np.reshape(data[i][j], (28, 28))
            image = axes[i][j].matshow(heatmap)
            fig.colorbar(image, ax=axes[i][j])
            axes[i][j].set_title(captions[i][j]), axes[i][j].set_xlabel('Weights'), axes[i][j].set_ylabel('Weights')

    plt.show()


class Network:
    def __init__(self, neurons_by_layer: list = None, activations_by_layer: list = None,
                 weight_file: str = None, momentum_alpha: float = 0.0, learning_rate = .0005):
        """
        make a list of lists of neurons
        """

        # default to reading from weight file to initialize neurons
        if weight_file is not None:
            weights = self.read_weights(weight_file)
            if activations_by_layer is None:
                activations_by_layer = [ActivationFunction.SIGMOID for _ in weights]
            self._neurons = [[Neuron(len(neuron) - 1, neuron[1:], neuron[0], learning_rate=learning_rate,
                                     momentum_alpha=momentum_alpha, activation_function=activation)
                              for neuron in weight_layer]
                             for activation, weight_layer in zip(activations_by_layer, weights)]
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
        with open(filename, 'w') as weight_file:
            for layer in self._neurons:
                for neuron in layer:
                    weight_file.write(
                        str(neuron.get_bias()) + ',' + ','.join([str(w) for w in neuron.get_weights()]) + '\n')
                weight_file.write('\n')

    def read_weights(self, filename: str) -> [[]]:
        with open(filename) as file:
            reader = csv.reader(file)
            layers = []
            layer_weights = []
            for row in reader:
                if len(row) == 0:
                    layers.append(layer_weights.copy())
                    layer_weights.clear()
                else:
                    layer_weights.append([float(weight) for weight in row])
        # layers.append(layer_weights)
        return layers

    def propagate_inputs(self, inputs):
        """
        propogate inputs and return final layer neurons
        :param inputs:
        :return:
        """
        input_list = [inputs]
        for layer in self._neurons:
            input_list.append([neuron.activate(input_list[-1]) for neuron in layer])
        return input_list

    def determine_error(self, label, results, confusion_matrix=None) -> float:
        return 0.0

    def determine_immediate_error(self, label, idx, output, data):
        # override this
        return 0.0

    def continue_training(self, current_error, desired_error, epoch, epoch_limit):
        if epoch_limit > 0:
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

    def train(self, training_file: str, epoch_size: int = 1000, desired_error_rate: float = .01, epoch_limit=-1):
        error_fractions = []
        error_fraction = 1
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
                errors += self.determine_error(label, result)
                total += 1

                # compute all the errors
                backprop_errors = [[self.determine_immediate_error(label, idx, output, result[0])
                                    * self._neurons[-1][idx].get_derivative_of_activation(result[-2])
                                    for (idx, output) in enumerate(result[-1])]]

                for i in range(len(result) - 3, -1, -1):  # iterates over layers
                    backprop_errors.insert(0, [self._neurons[i][idx].get_derivative_of_activation(result[i])
                                               * sum(self._neurons[i + 1][j].get_weights()[idx] * backprop_errors[0][j]
                                                     for j in range(len(self._neurons[i + 1])))
                                               for (idx, output) in enumerate(result[i + 1])])

                # apply the errors to all the weights
                for i in range(len(backprop_errors)):  # go layer by layer, updating weights
                    for j in range(len(self._neurons[i])):
                        self._neurons[i][j].update_weights(backprop_errors[i][j], result[i])

            error_fraction = errors / total
            if epoch % 10 == 0:
                print(epoch, error_fraction, time.time() - start, datetime.datetime.now().strftime("%H:%M:%S"))
                start = time.time()
                error_fractions.append(error_fraction)
            if epoch % 50 == 0:
                self.write_weights(f'weights/weights{time.time()}.csv')
            epoch += 1

        return error_fractions


class Classifier(Network):
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


class Autoencoder(Network):
    def determine_error(self, label, result: list, confusion_matrix: defaultdict = None):
        confusion_matrix[label][0] += .5 * sum((result[-1][i] - result[0][i]) ** 2 for i in range(len(result[0])))
        return .5 * sum((result[-1][i] - result[0][i]) ** 2 for i in range(len(result[0])))

    def determine_immediate_error(self, label, idx, output, data):
        return data[idx] - output


def problem1(hidden_neurons: int):
    network = Classifier(#weight_file='weights/weights1pt6pt.csv',
                         neurons_by_layer=[784, hidden_neurons, 10],
                         activations_by_layer=[ActivationFunction.RELU, ActivationFunction.LEAKY_RELU])
    initial_error_fraction, initial_classification = network.test('organizedData/1test')
    print(initial_error_fraction)

    error_fraction = network.train("organizedData/1training", epoch_size=1000, desired_error_rate=.005)
    # plot error fraction against epoch
    plt.plot([i * 10 for i in range(len(error_fraction))], error_fraction)
    plt.xlabel("Epoch")
    plt.ylabel("Error Fraction")
    plt.title("Error Fraction During Training of Network")
    plt.grid(True)
    plt.show()

    final_error_fraction, final_classification = network.test('organizedData/1test')
    print(final_error_fraction)
    print(final_classification)
    network.write_weights(f'weights/finalweights{final_error_fraction}_{hidden_neurons}.csv')

    return network


def problem2(hidden_neurons: int):
    network = Autoencoder(neurons_by_layer=[784, hidden_neurons, 784],
                          activations_by_layer=[ActivationFunction.RELU, ActivationFunction.LEAKY_RELU],
                          momentum_alpha=.001)
    initial_loss, initial_total_error = network.test('organizedData/1test')
    print('initial loss: ', initial_loss)

    error_fraction = network.train("organizedData/1training", epoch_size=1000, desired_error_rate=56)
    network.write_weights('weights/autoencoderweights.csv')
    # plot error fraction against epoch
    plt.plot([i * 10 for i in range(len(error_fraction))], error_fraction)
    plt.xlabel("Epoch")
    plt.ylabel("Error Fraction")
    plt.title("Error Fraction During Training of Network")
    plt.grid(True)
    plt.show()

    final_loss, final_total_error = network.test('organizedData/1test')
    print(final_loss)

    # plot initial and final loss
    metrics = ['Before and After']
    plt.bar(np.arange(len(metrics)) - 0.2, [initial_loss], 0.4, label='Initial')
    plt.bar(np.arange(len(metrics)) + 0.2, [final_loss], 0.4, label='Final')
    plt.xticks(np.arange(len(metrics)), metrics)
    plt.ylabel("J2 Loss Value"), plt.legend()
    plt.title("Loss of Autoencoder before and after Training")
    plt.show()

    # plot initial and final metrics
    split_metrics = [i for i in range(10)]
    plt.bar(np.arange(len(split_metrics)) - 0.2, [initial_total_error[i][0] / 1000 for i in initial_total_error], 0.4,
            label='Initial')
    plt.bar(np.arange(len(split_metrics)) + 0.2, [final_total_error[i][0] / 1000 for i in final_total_error], 0.4,
            label='Final')
    plt.xticks(np.arange(len(metrics)), metrics)
    plt.xlabel("Classes"), plt.ylabel("Average Loss"), plt.legend()
    plt.title("Average loss for each class")
    plt.show()

    return network


if __name__ == '__main__':
    # create_data('datafiles/MNISTnumImages5000_balanced.txt', 'datafiles/MNISTnumLabels5000_balanced.txt', 5000,
    #             {i for i in range(10)}, 100, 'organizedData/1')

    classifier = problem1(150)
    autoencoder = problem2(150)

    # select 20 hidden neurons to plot
    to_plot = set()
    while len(to_plot) < 20:
        to_plot.add(random.randint(0, len(classifier[0])))
    classifier_weights = [[], [], [], []]
    autoencoder_weights = [[], [], [], []]
    for index in to_plot:
        for i in range(len(classifier_weights)):
            if len(classifier_weights[i]) < 5:
                classifier_weights[i].append(classifier.read_weights('weights/weights1pt6pt.csv')[0][index][1:])
                autoencoder_weights[i].append(autoencoder.read_weights('weights/autoencoderweights.csv')[0][index][1:])

    plot_heatmaps(classifier_weights, [[f'Hidden Neuron {list(to_plot)[i*4+j]} Weights'
                                        for j in range(5)] for i in range(4)], 'Classifier Hidden Neuron Weights')
    plot_heatmaps(autoencoder_weights, [[f'Hidden Neuron {list(to_plot)[i*4+j]} Weights'
                                        for j in range(5)] for i in range(4)], 'Autoencoder Hidden Neuron Weights')

    to_plot = set()
    while len(to_plot) < 8:
        to_plot.add(random.randint(0, 1000))
    autoencoder_comparisons = [[], []]
    with open('organizedData/1test') as tests:
        for idx, test in enumerate(tests):
            if idx in to_plot:
                label, image = int(test.split(' ')[0]), [float(i) for i in test.strip().split(' ')[1:]]
                autoencoder_comparisons[0].append(image)
                autoencoder_comparisons[1].append(autoencoder.propagate_inputs(image)[-1])
    plot_heatmaps(autoencoder_comparisons, [[f'Input Image {list(to_plot)[i]}' for i in range(8)],
                                            [f'Output Image {list(to_plot)[i]}' for i in range(8)]], 'Autoencoder Before and After Sets')
