# Long Short-Term Memory Neural Networks

This is an implementation of LSTM networks in Python, using NumPy. You can access the relevant documents from below.

[Project report](report.pdf)

[Presentation poster](poster.pdf)

# How to use?

[lstm.py](lstm.py) is the stand alone working implementation of the network. You can call the script as follows in order to see available command line options.
```
python lstm.py -h
```
This variant of the network is adjusted to work with text files but you are free to modify it to work with any type of data. There is a dummy text file named [hello.txt](hello.txt), you shall provide your training file and replace that via using the *-input* flag. The rest of the parameters are completely optional but shall be adjusted according to your input file in order to get reasonable results.

The network consists of a single layer with 128 hidden units by default. The learning rate is set to be 0.1. After every 1000 iterations, a checkpoint file will be generated that records the learned weights and a 500 character long sequence will be generated via the so-far-trained network so you can observe how well the network performs. The parameters above are all adjustable using the command line flags.

There is also an experiment friendly GUI version of the network called [vis.py](vis.py). This script plots the weights of the hidden units in real time thus gives you an idea about how a hidden unit performs the act of *remembering*. You can also call the help option of this script in order to see all available options.
