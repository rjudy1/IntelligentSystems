import matplotlib.pyplot as plt
import numpy as np
import random


from neuron import Neuron, ActivationFunction
from network import Network
from sofm import SOFM


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


def plot_heatmaps(data: [[]], captions: [], image_caption: str, xlabel: str = '', ylabel: str = '', width=28,
                  height=28):
    # plot heatmaps of weights initial and final

    fig, axes = plt.subplots(nrows=len(data), ncols=len(data[0]))
    fig.suptitle(image_caption)
    fig.tight_layout()

    plt.subplots_adjust(left=0.05,
                        bottom=0.15,
                        right=0.95,
                        top=0.85,
                        wspace=0.02,
                        hspace=.2)

    # find max value in the first data value
    maximum = max((data[0][0])) + .05 * max((data[0][0]))
    minimum = min((data[0][0])) - .05 * max((data[0][0]))

    for i in range(len(data)):
        for j in range(len(data[0])):
            heatmap = np.reshape(data[i][j], (width, height))
            heatmap = np.transpose(heatmap)

            image = axes[i][j].matshow(heatmap, vmin=minimum, vmax=maximum)

            # Remove axis labels and ticks
            axes[i][j].tick_params(left=False, bottom=False, top=False, right=False, labelsize=0)
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

            axes[i][j].set_title(captions[i][j], fontsize=5)

    fig.subplots_adjust(right=.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(image, cax=cbar_ax)

    plt.show()


# _____________________________|__
# HW Specific code below here  |
# _____________________________v____

def problem1(hidden_neurons: int):
    # case I
    network1 = Network(weight_files=['weights/autoencoderweights_layer0.csv', None],
                       # weight_files=['weights/classifierI0.124_175.csv_layer0.csv', 'weights/classifierI0.124_175.csv_layer1.csv'],
                       neurons_by_layer=[784, hidden_neurons, 10], momentum_alpha=.025, learning_rate=.0001,
                       activations_by_layer=[ActivationFunction.RELU, ActivationFunction.LEAKY_RELU])
    # initial test
    initial_error_fraction, initial_classification = network1.test('organizedData/1test')
    print(f'initial error class I: {initial_error_fraction}')

    # train
    error_fraction1 = network1.train("organizedData/1training", epoch_size=1000, desired_error_rate=.005,
                                     epoch_limit=250, last_layer_only=True)
    error_fraction1.insert(0, initial_error_fraction)

    # final test and training classifications
    final_error_fraction, final_class = network1.test('organizedData/1test')
    print(final_error_fraction, [':'.join([str(i), str([final_class[i][j] for j in range(10)])]) for i in range(10)])
    _, final_training_classification = network1.test('organizedData/1training')
    print([':'.join([str(i), str([final_training_classification[i][j] for j in range(10)])]) for i in range(10)])

    network1.write_weights(f'weights/classifierI{final_error_fraction}_{hidden_neurons}')

    # case II
    network2 = Network(weight_files=['weights/autoencoderweights_layer0.csv', None],
                       # weight_files=['weights/classifierII0.043_175_layer0.csv', 'weights/classifierII0.043_175_layer1.csv'],
                       neurons_by_layer=[784, hidden_neurons, 10], momentum_alpha=.025, learning_rate=.001,
                       activations_by_layer=[ActivationFunction.RELU, ActivationFunction.LEAKY_RELU])
    initial_error_fraction, initial_classification = network2.test('organizedData/1test')
    print(initial_error_fraction)

    error_fraction2 = network2.train("organizedData/1training", epoch_size=1000, desired_error_rate=.005,
                                     epoch_limit=250, last_layer_only=False)
    error_fraction2.insert(0, initial_error_fraction)

    test_ef, final_class = network2.test('organizedData/1test')
    print(test_ef, [':'.join([str(i), str([final_class[i][j] for j in range(10)])]) for i in range(10)])
    _, final_training_classification = network2.test('organizedData/1training')
    print([':'.join([str(i), str([final_training_classification[i][j] for j in range(10)])]) for i in range(10)])

    network2.write_weights(f'weights/classifierII{test_ef}_{hidden_neurons}')

    # plot error fraction against epoch
    plt.plot([i * 10 for i in range(len(error_fraction1))], error_fraction1, label='Last Layer Training Only')
    plt.plot([i * 10 for i in range(len(error_fraction2))], error_fraction2, label='Full Network Training')
    plt.xlabel("Epoch")
    plt.ylabel("Error Fraction")
    plt.title("Error Fraction During Training of Network with Pretrained Input Layer Weights")
    plt.legend()
    plt.xlim([0, 250])
    plt.grid(True)
    plt.show()

    return network1, network2


def problem2():
    sofm = SOFM(weight_files=['weights/sofmconverged_layer0.csv'],
                neurons_by_layer=[784, 144], learning_rate=.01, width=12, height=12)
    # train feature map and converge
    sofm.train('organizedData/1training', epoch_size=2000, epoch_limit=1000, desired_error_rate=-1, curr_epoch=0)
    sofm.write_weights('weights/sofm')

    sofm.train('organizedData/1training', epoch_size=2000, epoch_limit=500, desired_error_rate=-1,
               convergence_stage=True)
    sofm.write_weights('weights/sofmconverged')

    error, activity_matrix = sofm.test('organizedData/1test')
    print(activity_matrix)
    print([':'.join([str(i), str([activity_matrix[i][j] for j in range(len(activity_matrix[i]))])]) for i in range(10)])

    activity_matrices = [[[0 if j not in activity_matrix[i] else activity_matrix[i][j] for j in range(144)]
                          for i in range(5)],
                         [[0 if j not in activity_matrix[i] else activity_matrix[i][j] for j in range(144)]
                          for i in range(5, 10)]]
    plot_heatmaps(activity_matrices, [[f'Map {i}' for i in range(5)],
                                      [f'Map {i}' for i in range(5, 10)]], 'Activity Maps for Each Class',
                  width=12, height=12)

    feature_map = [[sofm._neurons[0][j*12+i].get_weights() for j in range(12)] for i in range(12)]

    plot_heatmaps(feature_map, [[f'{i}, {j}' for j in range(12)] for i in range(12)],
                  'Features for entire feature map', width=28, height=28)
    return sofm


def problem3(sofmweights):
    # classifier using sofm as hiddden layer
    def func(inputs, layer):
        best_index = -1
        min_distance = 1000000000000000
        for idx, neuron in enumerate(layer):
            logit = sum((a-b)**2 for a, b in zip(inputs, neuron.get_weights()))
            if logit < min_distance:
                min_distance = logit
                best_index = idx
        return [0 if i != best_index else 1 for i in range(len(layer))]

    network1 = Network(weight_files=[sofmweights, None],
        #weight_files=['weights/weights1700235388_250_layer0.csv', 'weights/weights1700235388_250_layer1.csv'],
                       neurons_by_layer=[784, 144, 10], momentum_alpha=.015, learning_rate=.05,
                       activations_by_layer=[ActivationFunction.RELU, ActivationFunction.LEAKY_RELU],
                       special_activations=[func, None])
    initial_error_fraction, initial_classification = network1.test('organizedData/1test')
    print(f'initial error class I: {initial_error_fraction}')

    error_fraction1 = network1.train("organizedData/1training", epoch_size=1000, desired_error_rate=.005,
                                     epoch_limit=250, last_layer_only=True)
    print(f'error fraction 1: {error_fraction1}')
    network1.write_weights('weights/sofmclassifier_')

    # feature_map = [[network1._neurons[1][i*2+j].get_weights() for j in range(5)] for i in range(2)]
    #
    # plot_heatmaps(feature_map, [[f'{2*i+j}' for j in range(5)] for i in range(2)],
    #               'Features for entire feature map', width=12, height=12)

    # plot error fraction against epoch
    error_fraction1.insert(0, initial_error_fraction)
    # error_fraction1 = [.907, 0.195, 0.157, 0.164, 0.175, 0.169, 0.161, 0.164, 0.165, 0.15, 0.162, 0.173, 0.149, 0.147, 0.155, 0.159, 0.168, 0.178, 0.156, 0.169, 0.177, 0.154, 0.147, 0.154, 0.154, 0.145]
    plt.plot([i * 10 for i in range(len(error_fraction1))], error_fraction1, label='Last Layer Training Only')
    plt.xlabel("Epoch")
    plt.ylabel("Error Fraction")
    plt.title("Error Fraction During Training of Network with Pretrained Input Layer Weights")
    plt.legend()
    plt.xlim([0, 250])
    plt.grid(True)
    plt.show()

    final_error_fraction, final_classification = network1.test('organizedData/1test')
    print(final_error_fraction,
          [','.join([str(i), str([final_classification[i][j] for j in range(10)])]) for i in range(10)])
    final_test_errorfraction, final_training_classification = network1.test('organizedData/1training')
    print([','.join([str(i), str([final_training_classification[i][j] for j in range(10)])]) for i in range(10)])


if __name__ == '__main__':
    # create_data('datafiles/MNISTnumImages5000_balanced.txt', 'datafiles/MNISTnumLabels5000_balanced.txt', 5000,
    #             {i for i in range(10)}, 100, 'organizedData/1')

    classifier = problem1(175)
    sofm = problem2()
    problem3('weights/sofmconverged_layer0.csv')
