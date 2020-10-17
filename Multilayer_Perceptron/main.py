from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mlp import MultiLayerPerceptron

def main():
    print("In Main .. Starting ..")
    
    #Generating Data with 200 samples  containing 5 features and two traget classes
    data = make_classification(n_samples=200, n_features=5, n_classes=2)
        
    #Spliting data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(data)

    #Normalizing the data in range [0,1]
    X_train = X_train/X_train.max()
    X_test = X_test/X_test.max()

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
if __name__ == "__main__":
    main()