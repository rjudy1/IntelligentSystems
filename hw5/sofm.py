from collections import defaultdict
import datetime
import math
import random
import time

from network import Network


class SOFM(Network):
    def __init__(self, neurons_by_layer: list, activations_by_layer: list = None, weight_files: [str] = None,
                 learning_rate=.05, width=12, height=12):
        """
        make a list of lists of neurons

        expect weight_files, if not None, to have a list of files, one per layer
        """

        # default to reading from weight file to initialize neurons
        super().__init__(neurons_by_layer, activations_by_layer, weight_files, 0, learning_rate)
        self._width = width
        self._height = height

    def get_euclidean_distance(self, index1, index2):
        x1, x2 = index1 // self._width, index2 // self._width
        y1, y2 = index1 % self._width, index2 % self._width
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def propagate_inputs(self, inputs):
        """
        propogate inputs and return final layer neurons
        :param inputs:
        :return:
        """
        best_index = -1
        min_distance = 1000000000000000
        for idx, neuron in enumerate(self._neurons[0]):
            if sum((a-b)**2 for a, b in zip(inputs, neuron.get_weights())) < min_distance:
                min_distance = sum((a-b)**2 for a, b in zip(inputs, neuron.get_weights()))
                best_index = idx
        return best_index

    def determine_error(self, label, result: int, confusion_matrix: defaultdict = None):
        confusion_matrix[label][result] += 1
        return 0

    def train(self, training_file: str, epoch_size: int = 1000, desired_error_rate: float = .01, epoch_limit=-1,
              last_layer_only=False):
        error_fraction = desired_error_rate * 1000
        epoch = 0
        with open(training_file) as file:
            trainings = file.readlines()
        start = time.time()
        while self.continue_training(error_fraction, desired_error_rate, epoch, epoch_limit):
            random.shuffle(trainings)
            for test in trainings[:epoch_size]:
                label, image = int(test.split(' ')[0]), [float(i) for i in test.strip().split(' ')[1:]]

                # test image in neuron, adjust weights
                ri = self.propagate_inputs(image)
                sigma0 = .5 * math.sqrt(self._height**2 + self._width**2)
                for r, neuron in enumerate(self._neurons[0]):
                    neighborhood = math.e ** (-self.get_euclidean_distance(r, ri)**2 / (2 * sigma0 * math.e ** (-epoch / epoch_limit)))
                    neuron.update_weights(neighborhood, [(a-b) for a, b in zip(image, neuron.get_weights())])

            if epoch % 10 == 0:
                print(epoch, error_fraction, time.time() - start, datetime.datetime.now().strftime("%H:%M:%S"))
                start = time.time()
            if epoch % 20 == 0:
                self.write_weights(f'weights/sofmweights{time.time()}')
            epoch += 1

        return None
