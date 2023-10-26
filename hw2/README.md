# HW 2 Python File
This code contains a function for a generic neuron that is used to simulated regular spiking and fast spiking neurons
and compare their mean spike rates over a variety of injected currents. It is then compared to ChatGPT-generated code
for a regular spiking neuron.

## Dependencies
- python3
- matplotlib
- numpy

## Usage
In order to run the code provided, simply run `python hw2.py`. This will first simulate the regular spiking neuron, 
then then the fast spiking neuron, and finally the ChatGPT generated code. Plots will appear for the time series of 
membrane potentials as well as the mean spike rates as compared to the injected currents.

The `neuron` function can also be independently called if other types of neurons should be simulated.