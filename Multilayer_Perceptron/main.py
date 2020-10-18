from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mlp import MultiLayerPerceptron
from dense import Dense

def main():
    print("In Main .. Starting ..")
    
    #Generating Data with 200 samples  containing 5 features and two traget classes
    data = make_classification(n_samples=200, n_features=5, n_classes=2)

    #Spliting data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1])
    
    #Normalizing the data in range [0,1]
    X_train = X_train/X_train.max()
    X_test = X_test/X_test.max()

    #-------------------------------Experimental Code-----------------------------------------
    
    print('-------------------------------------------------------------------')
    print("Layer: Dense1: ")
    print("-----------------------Forward Pass Layer 1------------------------")

    dense1  = Dense(5, 8)
    print('Inputs Shape  : ', X_train[0].shape)
    print('Weights Shape : ', dense1.weights.shape)
    dense1.forward_pass(X_train[0])

    print("Max in act_pot: ", dense1.activation_potentials.max())
    print("Min in act_pot: ", dense1.activation_potentials.min())
    print()
    print("Max in outputs: ", dense1.outputs.max())
    print("Min in outputs: ", dense1.outputs.min())
    print("-----------------------Forward Pass Layer 1------------------------")
    print('-------------------------------------------------------------------')
    print("Layer: Dense2: ")
    print("-----------------------Forward Pass Layer 2------------------------")

    dense2 = Dense(8, 2)
    print("Input Shape   : ", dense1.outputs.shape)
    print("Weights Shape : ", dense2.weights.shape)
    dense2.forward_pass(dense1.outputs)

    print("Max in act_pot: ", dense2.activation_potentials.max())
    print("Min in act_pot: ", dense2.activation_potentials.min())
    print()
    print("Max in outputs: ", dense2.outputs.max())
    print("Min in outputs: ", dense2.outputs.min())
    print("-----------------------Forward Pass Layer 2-------------------------")
    print('--------------------------------------------------------------------')
    print('Layer: Dense2: ')
    print("-----------------------Backward Pass Layer 2------------------------")
    print("-----------------------Backward Pass Layer 2------------------------")
    print('--------------------------------------------------------------------')
    print("Layer: Dense1: ")
    print("-----------------------Backward Pass Layer 1------------------------")
    print("-----------------------Backward Pass Layer 1------------------------")


    #-------------------------------Experimental Code-----------------------------------------
    '''
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