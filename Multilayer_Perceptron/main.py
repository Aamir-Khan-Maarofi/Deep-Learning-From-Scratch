from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mlp import MultiLayerPerceptron
from dense import Dense

def main():
    print("In Main .. Starting ..")
    
    #Generating Data with 200 samples containing 5 features and two traget classes
    data = make_classification(n_samples=200, n_features=5, n_classes=2)

    #Spliting data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1])
    
    #Normalizing the data in range [0,1]
    X_train = X_train/X_train.max()
    X_test = X_test/X_test.max()

    #-------------------------------Experimental Code-----------------------------------------
    network = []
    print("----------------------------Layer 1-------------------------")
    network.append(Dense(5,8))
    print("----------------------------Layer 2-------------------------")
    network.append(Dense(8,2))
    
    c = 1
    print(network[0].weights)
    print(network[1].weights)
    for x,y in zip(X_train, y_train):
        print("      --------------- Training Sample {}-----------------".format(c))
        #Keep track of inputs 
        previous_inputs = []    
        previous_inputs.append(x)
        
        for layer in network:
            #Forward Pass
            previous_inputs.append(layer.forward_pass(previous_inputs[-1]))
        
        #Ignore the output of last layer
        previous_inputs = previous_inputs[:-1]

        for layer, inp in zip(network[::-1], previous_inputs[::-1]):
            #Backward Pass
            if layer != network[-1]:
                layer.backward_pass(0.1, inp, y, layer, network, prev_loc_gradients, prev_weights)
            else:
                layer.backward_pass(0.1, inp, y, layer, network)
            prev_loc_gradients = layer.loc_gradients
            prev_weights = layer.weights
        c += 1
    print(network[0].weights)
    print(network[1].weights)
'''
    print('-------------------------------------------------------------------')
    print("Layer: Dense1: ")
    print("-----------------------Forward Pass Layer 1------------------------")

    
    print('Inputs Shape  : ', X_train[0].shape)
    print('Weights Shape : ', dense1.weights.shape)
    #dense1.forward_pass(X_train[0])
    print('Output Shape  : ', dense1.outputs.shape)

    print("Max in act_pot: ", dense1.activation_potentials.max())
    print("Min in act_pot: ", dense1.activation_potentials.min())
    print()
    print("Max in outputs: ", dense1.outputs.max())
    print("Min in outputs: ", dense1.outputs.min())
    print("-----------------------Forward Pass Layer 1------------------------")
    print('-------------------------------------------------------------------')
    
    print("Layer: Dense2: ")
    print("-----------------------Forward Pass Layer 2------------------------")
    print("Input Shape   : ", dense1.outputs.shape)
    print("Weights Shape : ", dense2.weights.shape)
    #dense2.forward_pass(dense1.outputs)
    print('Outputs Shape : ', dense2.outputs.shape)
    print("Max in act_pot: ", dense2.activation_potentials.max())
    print("Min in act_pot: ", dense2.activation_potentials.min())
    print()
    print("Max in outputs: ", dense2.outputs.max())
    print("Min in outputs: ", dense2.outputs.min())
    print("-----------------------Forward Pass Layer 2-------------------------")
    print('--------------------------------------------------------------------')
    print('Layer: Dense2: ')
    print("-----------------------Backward Pass Layer 2------------------------")
    print('Y_train 1st entry: ', y_train[0])
    dense2.backward_pass(0.1, dense1.outputs ,y_train[0], dense2, network)
    print("-----------------------Backward Pass Layer 2------------------------")
    print('--------------------------------------------------------------------')
    print("Layer: Dense1: ")
    print("-----------------------Backward Pass Layer 1------------------------")
    dense1.backward_pass(0.1, X_train[0], y_train[0], dense1, network, dense2.loc_gradients, dense2.weights)
    print("-----------------------Backward Pass Layer 1------------------------")


    #-------------------------------Experimental Code-----------------------------------------



    #Creating nework 
    MLP = MultiLayerPerceptron()

    #Training network
    #MLP.train()

    #Testing network
    #MLP.test()

    #Validating network
    #MLP.validate()

    #New Predictions
    #MLP.predict()

'''
if __name__ == "__main__":
    main()