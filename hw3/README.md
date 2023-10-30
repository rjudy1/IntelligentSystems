# HW 3 Python File
The code contains a function to split the provided MNIST data set into a training, test, and challenge set as well as 
a Perceptron class that implements tests, challenges, and training functionality for a perceptron. When run, it creates
and trains two perceptrons for differentiating between 0s and 1s v 0s and 9s and generates plots representing the training
and performance.

## Dependencies
- python3
- csv
- datetime
- enum
- matplotlib
- numpy
- random
- time

## Usage
In order to run the code provided, simply run `python network.py`. This will simulate the network training and testing
for both the classifier and autoencoder. The weight file argument for the Network declarations can be commented out to 
train from scratch instead of reading from the preserved files.

Changing parameters can be done by editing below the comment for problem specfic code in network.py. The Network and Neuron
classes are meant to be reusable.