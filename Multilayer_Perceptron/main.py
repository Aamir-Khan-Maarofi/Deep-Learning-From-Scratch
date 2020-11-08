from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize

import matplotlib.pyplot as plt
from dense import Dense
import numpy as np
import random


def main():
    print("In Main .. Starting ..")

    # Generating Data with 10 samples containing 4 features, one target calss
    # The target in binary either 0 or 1
    data = make_classification(
        n_samples=100, n_features=4, n_informative=4, n_redundant=0, n_classes=2)

    # Spliting training features and training targets
    X_train = data[0]
    y_train = data[1]

    # Normalizing the data in range [0,1]
    X_train = normalize(X_train)

    plt.plot(X_train)
    plt.show()

    print('Training Data: (X)   : ', X_train)
    print('Taraining Data: (y)  : ', y_train)

    # Encoding data using one hot encoding technique
    enc = OneHotEncoder(sparse=False)
    desired = enc.fit_transform(y_train.reshape(-1, 1))
    print("One Hot encoded: (y) : ", desired)

    # -------------------------------Experimental Code-----------------------------------------
    # netwok is a list of all layers in the network, each time a layer is created as Dense() object, will be appended to network
    network = []
    print("----------------------------Layer 1-------------------------")
    network.append(Dense(4, 6))
    print('Created Only Hidden Layer: N_nodes: {} , N_inputs: {}'.format(6, 4))

    print("----------------------------Layer 2-------------------------")
    network.append(Dense(6, 2))
    print('Created Output Layer: N_nodes: {} , N_inputs: {}'.format(2, 6))

    # List to store the total error
    error = []

    # initialized epoch to 1
    epoch = 1
    # The main training loop, exits at after 10 epochs
    while epoch <= 100:

        # Random suffling of training data
        temp = list(zip(X_train, y_train))
        random.shuffle(temp)
        X_train, y_train = zip(*temp)

        print('--------------------------Epoch: {}-------------------------'.format(epoch), '\n')

        # Select one feature vector from feature matrix and corresponding target vector from desired matrix (Which was obtained from one hot encoding)
        for x, y in zip(X_train, desired):

            # previous_inputs list keeps track of inputs from layer to layer in each epoch
            previous_inputs = []

            # At start of each epoch, the list contains only inputs from input nodes which are the features of current training sample
            previous_inputs.append(x)

            # This loop iterates over all layers in network and perform forward pass
            for layer in network:
                # Forward_pass perform forward propagation of a layer of last element of the previous_inputs list,
                # and returns the output of layer which is stored as ndarray in list, as it will be used as inputs to next layer
                previous_inputs.append(layer.forward_pass(previous_inputs[-1]))

            # Ignore the output of last layer, as I'm using the preious_inputs array in reverse order in backward_pass in next loop
            previous_inputs = previous_inputs[:-1]

            # Next loop reverses the network and previous_inputs lists to perform  backward propagation of al layers from output layer all the way to input layer
            for layer, prev_inputs in zip(network[::-1], previous_inputs[::-1]):

                # If the layer is not output layer then perform backward propagation using code inside if statement
                if layer != network[-1]:

                    # call to backward_pass using learning rate = 0.0001, inputs to current layer, target vector 'y',
                    # previous_loc_gradients (local gradients of layer next to current layer),
                    # and prev_weights (weights of layer next to current layer)
                    # Store the updated weights and biases for mean square error calculation at end of epoch
                    layer.backward_pass(
                        0.1, prev_inputs, y, prev_loc_gradients, prev_weights)

                # otherwise, perform the backward pass for output layer using code in else block
                else:
                    layer.backward_pass(
                        0.1, prev_inputs, y)

                # Store local gradients and weights of current layer for next layer backward pass
                prev_loc_gradients = layer.loc_gradients
                prev_weights = layer.weights

            # error_i is sum of errors for all training examples on updated weights and biases
            error_i = 0

            # This loop calculates Total Error on new weights and biases, by considering the whole training data
            for x_val, y_val in zip(X_train, desired):
                previous_inputs = []
                previous_inputs.append(x_val)

                # Perform  forward pass on new weights and biases
                for layer in network:
                    # Forward Pass
                    previous_inputs.append(
                        layer.forward_pass(previous_inputs[-1]))

                # add the error of prediction of current training sample to pervious errors
                error_i += np.power((previous_inputs[-1] - y_val), 2).sum()

        # Append total error of current sample to error list, and repeat the process for next sample, do this for all samples
        error.append(error_i)
        print("-----------------")
        print("Error: ", error_i)
        print("-----------------")
        print()

        if error[epoch - 2] > error_i:
            print('Decreasing...')
        elif error[epoch - 2] == error_i:
            print('Nothing is happening...')
        else:
            print('Increasing...')
        # Increase epoch by one, and perform forward, backward on next sample, and calculate error for all samples, do this until while  is true
        epoch += 1

    # Plot the errors after training completes,
    plt.plot(error, color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Total Error Graph')
    plt.show()
    # -------------------------------Experimental Code-----------------------------------------


if __name__ == "__main__":
    main()
