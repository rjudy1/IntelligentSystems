# HW 5 Python Files
The code for HW5 consists of `network.py`, `sofm.py` and `neuron.py`. neuron.py remains unchanged from HW4.
The Network class remains largely consistent with that of HW4. A few changes were made to the `network.py` in
order to better read in individual layers from files as well as to insert a special activation function for the 
self-organized feature map.

The file `sofm.py` contains a class derived from the Network class that uses a similar Nueron structure to create
and train a self-organized feature map. This consists of changing the meaning of the output of the layer and adds
a function to find euclidean distance between points in the map.

Additionally, the `main.py` file contains modified versions of the functions to generate test and training data, 
to plot heatmaps for the various neurons, and to initialize, train, and test the different networks for this homework.

## Python3 Dependencies
- datetime
- enum
- math
- matplotlib
- numpy
- random
- 

## Usage
`network.py` has been minorly changed to account for substituting layers with special activation functions
and individual weight files. Thus, it was included in the code. Copy `neuron.py` to the folder containing the
`main.py`, `network.py`, and `sofm.py`.

In order to run the code provided, simply run `python main.py`. This will run the code for the three 
problems and show the relevant plots. Note that this expects the data to be copied into the datafiles
folder as well and then the `create_data` function to be used. The values in the `weight_files` argument
can be changed to use the weights in files saved by the `Network` class or can be randomized by setting
the array to an array of `None`s.