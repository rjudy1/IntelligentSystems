# author: Rachael Judy
# date: 17 October 2023
# purpose: simulate a perceptron to differentiate between two digits from the MNIST set

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random


def create_data(values: set, training_quantity: int, test_quantity: int, prefix: str) -> None:
    """
    This creates three files, a challenge, test, and training set for the values to classify from an assumed 500 values
    per item

    :param values: values to sample from for classifying
    :param training_quantity: number of values to select from each class
    :param test_quantity: number of values to select from each class for the test set and challenge sets
    :param prefix: prefix for output filenames
    """

    # select test_quantity indices to put in the test and challenge set
    test_indices = set()
    while len(test_indices) < test_quantity:
        test_indices.add(random.randint(0, 499))

    training_data = list()
    test_file = open(''.join([prefix, 'test']), 'w')
    challenge_file = open(''.join([prefix, 'challenge']), 'w')
    with open("datafiles/MNISTnumImages5000_balanced.txt") as images, open("datafiles/MNISTnumLabels5000_balanced.txt") as labels:
        for index, (image, label) in enumerate(zip(images, labels)):
            if index % 500 not in test_indices and int(label.strip()) in values:
                training_data.append(' '.join([label.strip(), image.strip().replace('\t', ' '), '\n']))
            elif index % 500 in test_indices and int(label.strip()) in values:
                test_file.write(' '.join([label.strip(), image.strip().replace('\t', ' '), '\n']))
            elif index % 500 in test_indices:
                challenge_file.write(' '.join([label.strip(), image.strip().replace('\t', ' '), '\n']))

    training_file = open(''.join([prefix, 'training']), 'w')
    random.shuffle(training_data)
    training_file.writelines(training_data)
    training_file.close()
    test_file.close()
    challenge_file.close()


class Perceptron:
    def __init__(self, input_size: int, initial_weights: list = None,
                 bias: float = random.random()/2, learning_rate: float = .0005):
        self._weights = initial_weights if initial_weights\
            else [random.random() / 2 for _ in range(input_size)]
        self._bias = bias
        self._learning_rate = learning_rate

    def get_weights(self) -> list:
        return self._weights

    def get_bias(self) -> float:
        return self._bias

    def set_bias(self, bias):
        self._bias = bias

    def activate(self, inputs) -> int:
        return 1 if sum(i[0] * i[1] for i in zip(inputs, self._weights)) + self._bias > 0 else 0

    def test(self, test_file: str) -> (float, float, float, float):
        true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
        with open(test_file) as tests:
            for test in tests:
                label, image = int(test.split(' ')[0] != '0'), [float(i) for i in test.strip().split(' ')[1:]]
                result = self.activate(image)
                if result == 1 == label:
                    true_positives += 1
                elif result == 0 == label:
                    true_negatives += 1
                elif result == 1:
                    false_positives += 1
                else:
                    false_negatives += 1

        recall = true_positives / (true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        f1 = 2 * precision * recall / (precision + recall)
        error_fraction = (false_positives + false_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)
        return error_fraction, precision, recall, f1, specificity

    def challenge(self, challenge_file: str) -> dict:
        result_map = defaultdict(lambda: [0, 0])
        with open(challenge_file) as tests:
            for test in tests:
                label, image = int(test.split(' ')[0]), [float(i) for i in test.strip().split(' ')[1:]]
                result_map[label][self.activate(image)] += 1
        return result_map

    def train(self, training_file: str, desired_error_rate: int = .005):
        error_fractions = []
        error_fraction = 1
        for i in range(32):
        # while error_fraction > desired_error_rate:  # replace the above line with this line to
                                                        # train to a rate instead of a set number
            with open(training_file) as trainings:
                errors, total = 0, 0
                for test in trainings:
                    label, image = int(test.split(' ')[0] != '0'), [float(i) for i in test.strip().split(' ')[1:]]

                    # test image in neuron, adjust weights
                    result = self.activate(image)
                    if result != label:
                        errors += 1
                    total += 1
                    for i in range(len(self._weights)):
                        self._weights[i] += self._learning_rate * (label - result) * image[i]
                    self._bias += self._learning_rate * (label - result)

                error_fraction = errors / total
                error_fractions.append(error_fraction)
        return error_fractions


def differentiate(data: set):
    create_data(data, 400, 100, 'organizedData/1')
    perceptron_1 = Perceptron(784)
    initial_weights = perceptron_1.get_weights().copy()
    error_fraction_initial, precision_initial, recall_initial, f1_initial, specificity_initial \
        = perceptron_1.test('organizedData/1test')
    error_fractions = perceptron_1.train('organizedData/1training')
    error_fraction_final, precision_final, recall_final, f1_final, specificity_final\
        = perceptron_1.test('organizedData/1test')

    # plot error fraction against epoch
    error_fractions.insert(0, error_fraction_initial)
    plt.plot(error_fractions)
    plt.xlabel("Epoch")
    plt.ylabel("Error Fraction")
    plt.title("Error Fraction During Training of Perceptron")
    plt.grid(True)
    plt.show()

    # plot initial and final metrics
    metrics = ['Error Fraction', 'Precision', 'Recall', 'F1 Score']
    plt.bar(np.arange(len(metrics)) - 0.2, [error_fraction_initial, precision_initial,
                                            recall_initial, f1_initial], 0.4, label='Initial')
    plt.bar(np.arange(len(metrics)) + 0.2, [error_fraction_final, precision_final,
                                            recall_final, f1_final], 0.4, label='Final')
    plt.xticks(np.arange(len(metrics)), metrics)
    plt.xlabel("Metrics"), plt.ylabel("Metric Value"), plt.legend()
    plt.title("Metrics for Untrained v Trained Perceptron")
    plt.show()

    # test different weights and plot
    error_fractions, precisions, recalls, f1s, specificities = [], [], [], [], []

    print(perceptron_1.get_bias())
    for w0 in np.linspace(min(perceptron_1.get_bias() * .8, perceptron_1.get_bias() * 1.2),
                          max(perceptron_1.get_bias() * .8, perceptron_1.get_bias() * 1.2), 21):
        perceptron_1.set_bias(w0)
        error_fraction, precision, recall, f1, specificity, = perceptron_1.test('organizedData/1test')
        error_fractions.append(error_fraction)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        specificities.append(specificity)
    metrics = ['Error Fraction', 'Precision', 'Recall', 'F1 Score']
    for index, values in enumerate(zip(error_fractions, precisions, recalls, f1s)):
        plt.bar(np.arange(len(metrics)) - (10-index) * 0.03, values, 0.06,
                label=f'{np.linspace(perceptron_1.get_bias() * .8, perceptron_1.get_bias() * 1.2, 21)[index]}')

    plt.xticks(np.arange(len(metrics)), metrics)
    plt.xlabel("Metrics"), plt.ylabel("Metric Value")
    plt.title("Metrics for Varying Biases")
    plt.show()

    # plot roc curve
    for i in range(len(specificities)):
        specificities[i] = 1 - specificities[i]
    plt.plot(specificities, recalls)
    plt.xlabel("1 - Specificities"), plt.ylabel("Sensitivity")
    plt.title("ROC Curve"), plt.grid(True)
    # print(min(specificities), max(recalls), specificities[10], recalls[10])
    plt.show()

    # plot heatmaps of weights initial and final
    initial_heatmap = np.reshape(initial_weights, (28, 28))
    final_heatmap = np.reshape(perceptron_1.get_weights(), (28, 28))
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    ax1, ax2 = axes
    im1 = ax1.matshow(initial_heatmap)
    im2 = ax2.matshow(final_heatmap)
    fig.colorbar(im1, ax=ax1), fig.colorbar(im2, ax=ax2)
    ax1.set_ylabel('Pixels'), ax1.set_xlabel('Pixels')
    ax2.set_ylabel('Pixels'), ax2.set_xlabel('Pixels')
    ax1.set_title('Initial Weights')
    ax2.set_title('Final Weights')
    plt.show()

    # present challenge dataset, output results
    print(f"Challenge results: {perceptron_1.challenge('organizedData/1challenge')}")


if __name__ == '__main__':
    # differentiate({0, 1})
    differentiate({0, 9})
