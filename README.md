# About

The code in this repository is used to experiment with the Multi-Layer Perceprton model on the features generated by the Local Binary Pattern algorithm on cifar-10 database. From the features perspective, nothing other than the simple generation of a histogram based on the results obtained from the LBP algorith is done, as adding multiple, complex features is not the aim of this experiment.

In order to run the code, simply run the `app.py` script. The script will require you to have the dataset (extract the `cifar-10-batches.py` directory) in the same directory in order to unpack the batches. After running, you will be left with those files:
  - test_data
  - train_data

Those files contain the observations required for the algorithm to train and test.

## Project Structure

`app.py` is the main script, all processes are called through it. `lbp.py` calls `load_data.py` to load the data from the dataset (specifically, the `cifar-10-batches-py` directory) and then generate features according to the histograms generated from image regions.
  
 The database can be downloaded from [the official cifar-10 web page](https://www.cs.toronto.edu/~kriz/cifar.html).

`train_data` and `test_data` are pickle files containing DataFrames with features for training and testing. Their main purpose is to serve as a storing method in order to spare some time from the total runtime (LBP histograms are quite computationally expensive).
