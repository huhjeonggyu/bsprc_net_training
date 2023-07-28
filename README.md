# bsprc_net_training (version 0.0.1)

The Monte-Carlo simulation to generate a large amount of time value data for arbitrary model parameters and option variables, where the time value means 'price-payoff' of the option. Afterward, we train a neural network to best learn this simulated data, with the goal of essentially considering the network as a closed pricing formula for the option. This approach potentially offers feasibility for all option pricing models. The current implementation supports both CPU and GPU for data generation, while network training is carried out solely on the GPU. Additionally, to complement the parallel processing capability of the CPU, 10 multi-processes are running. Parallel processing operations may be faster on the GPU, but the CPU might be faster if not.

The important notebook (.ipynb) files are as follows:
1. sample.ipynb: data generation
2. train.ipynb: network training
3. test.ipynb: network performance test

Other Python (.py) files are as follows:
1. model.py: network definition
2. utils.py: definition of various utility functions

There are also two directories with the following roles:
1. data: This is where the generated data is stored. Training files are in data/train, while test files are stored in data/test.
2. net: This is where the trained network is stored. The file name of the network includes the number of data used for training.

Additionally, the following facts are important.
1. Each data file contains 10,000 values.
2. The range for model parameters and option variables is inclusive of volatility sigma from 0.01 to 1, maturity T from 0.01 to 1, and strike price K such that log(K)/sqrt(T) ranges from -2 to 2.
3. The network is a simple MLP with two hidden layers of 1000 nodes each, accepting sigma, T, K, and outputting the option's time value tv.
4. The network is trained using the ADAM optimizer, and 30% of the training data is used solely for checking the learning rate decay condition.
5. The early stopping rule applies if the loss of the validation dataset no longer decreases. (Deciding when to end training is a very important issue!)
6. If the experiment goes well, as the number of data increases tenfold, R2 in test.ipynb is expected to roughly increase tenfold, and MSE is expected to decrease roughly tenfold.

## Usage
1. sample.ipynb: Remember that each data generation saves 10,000 data.
* train_data_num: number of training data files
* test_data_num: number of test data files
2. train.ipynb: network training
* data_num: Decide how many data to use for network training. (Multiple of 10,000)
3. test.ipynb: network performance test
* train_data_num: Decide how much data to test the trained network. (Multiple of 10,000)
* test_data_num: Set the number of test data. Try to use all the generated test data. (Multiple of 10,000)
