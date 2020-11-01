'''
dense.py module contains only class Dense which is implementaion 
of a layer in fully connected multilayer perceptron
'''


#Importing numpy for vector operations
import numpy as np

#Setting random seed to generate similar weights an biases vector in each execution
#I set the seed for random, just to inspect the current and updated weights
np.random.seed(78)


class Dense():
    """ 
    Dense represent a layer in fully connected multilayer perceptron network
    ...

    Attributes
    ----------
    weights : 2D array i.e. matrix, shape = (n_inputs, n_nodes)
        Stores the weights of all neurons in the layer
    outputs : 1D vector of outputs, shape = (n_nodes,)
        Stores the outputs of all neurons in the layer
    biases :  1D vector, shape = (n_nodes,)
        Stores the biases of all neurons in the layer, initialy set to one
    activation_potentials : 1D vector, shape = (n_nodes,)
        Store the  activation_potentials of all the neurons in the layer
    local_grads : 1D vector, shape = (n_nodes,)
        Store local gradients of all neurons in the layer
    
    Methods 
    -------
    Constructor(n_inputs, n_nodes) :
        Initialzes weights and biases of the network  
    sigmoid(inputs) : 
        Activates the activation_potential -> ndarray (act_potentials of all neurons in layer)
        Save the activated ndarry to outputs -> ndarray (outputs of all neurons in this layer) 
    d_sigmoid() :
        Returns the first order differential value of the activation function
    forward_pass() : 
        Computes the activation potentials of the current layer neurons, which is 
        inputs * weights + biases 
        Call to activation function and returns the outputs
    backward_pass() : 
        Calculate the local gradients of all neurons in the layer, update weights and biases of
        all neurons in current layer based on delta rule:
        w(n + 1) = w(n) + learning_rate * local gradient * input_to_neuron
    """

    def __init__(self, n_inputs, n_nodes):
        '''
        Constructor : Initialze weights and biases arrays, based on 
        n_inputs    : Number of inputs to this layer
        n_nodes     : Number of nodes in this layer
        '''
        
        #Weights associated with all n_nodes for (n_inputs) synaptic inputs, shape = (n_inputs, n_nodes)
        #Each column in the weight matrix is weight vector associated with one neuron
        self.weights = np.random.uniform(low=0, high=1, size=n_inputs*n_nodes).reshape(n_inputs, n_nodes)
        
        #Biases associated to each neuron, shape (n_nodes,)
        #There are n_nodes elements and each one represent bias associated with one neuron in layer
        #This one dimension array will be added to linear cobinations of all neurons  
        #assuming the synaptic connection associated to biases is conceptually '1'
        self.biases = np.ones(n_nodes)

    def sigmoid(self):
        '''
        sigmoid() : activation function, to squash the output of each neuron between 0 and 1
        params    : No Parameters
        Return    : The activated outputs shape (n_nodes,) of all neurons in the layer
        Operation : Uses sigmoid function which operates on all activation potentials of the layer
        '''

        #Activates the activation_potential -> ndarray
        #Save the activated ndarry to outputs
        self.outputs = 1 / (1 + np.exp(-self.activation_potentials))
  
        return self.outputs

    def d_sigmoid(self):
        '''
        d_sigmoid : First order differential of sigmoid activation function
        params    : No parameters
        Return    : Vector containing values of the first order differential of sigmoid
        Operation : Takes outputs of layer and calculates the values of first order 
                    differential of activation function, which will be used in local gradient calculation
        '''

        # Vector value of shape (n_nodes,) 
        return self.outputs * (1 - self.outputs)
        
    def forward_pass(self, inputs):
        '''
        forward_pass : Forward pass of current layer for inputs
        params       : inputs -> vector value, one training sample at a time
        returns      : output vector of layer 
        operations   : Find activation potentials, call activation function and return outputs   
        '''
        
        #Calculate activation potential of the current layer neurons -> ndarray 
        #Which is inputs times weights add a bias and stores it to activation_potentials
        self.activation_potentials = np.dot(self.weights.T, inputs) + self.biases
        
        return self.sigmoid()

    def backward_pass(self, learning_rate, inputs_to_layer, target,
                      prev_loc_grads = [], prev_weights = []):
        '''
        backward_pass : Backward pass of current layer
        Note          : Backward signal origninates at output node, propegate all the way to 
                        input node, "Previous_loc_grads and Previous_weights are local gradients 
                        and weights associated with a layer next to current layer which made call
                        to backward_pass method. They are set keyword arguments thus when no value
                        is passed for prev_loc_grads and prev_weights they will contain reference 
                        to empty lists. This specifies the case of call to backward pass with output
                        layer,  hence there is no layer preceeding output layer, so no local gradients
                        and weights will be passed when call to this method is made with output layer,               
        
        Params        : learning_rate   -> Step size of gradient descent
                        inputs_to_layer -> Synaptic inputs of the layer
                        target          -> One hot encoded output corresponding to training sample
                                           used in forward pass, need this for error computation 
                        prev_loc_grads  -> Local gradient of layer next to current layer
                        prev_weights    -> Synaptic weights of layer next to current layer
        
        Returns       : Updated weights and biases of current layer, that will be used in total error computation

        Operations    : Confirms if previous local gradients are not passed then the call is made 
                        with output layer, in this case, computes relative error of output neurons
                        based on targets (one hot encoded) and outputs of the neurons. and then compputes
                        the local gradients of output neurons. Otherwise, computes the local gradient of 
                        neurons.

                        Once local gradients are calculated, this method updates weights and biases, and 
                        return the updated weights and biases to caller.  
        '''
        
        #If prev_loc_grads length is '0' then if is executed, which indicates that no local gradients
        #are passed, this is the case when call is made  with output layer as no layer preceeds output layer
        if not len(prev_loc_grads):
            
            #For output layer, firstly relative error is calculated,
            #error_at_end_nodes is vector of shape (n_nodes,), same is target, their element-wise
            #subtraction results in vector of same dimensions and size. 
            self.error_at_end_nodes = self.outputs - target
            
            #Secondly, uses the computed error and d_sigmoid() to calculate local gradients
            #Based on formula: local gradients at output = e_k * derivitive of sigmoid
            self.loc_gradients = self.error_at_end_nodes * self.d_sigmoid()
        else:
            # again, previous here refer to layer prceeding current layer (layer next to current)
            # Hidden layers local gradients => derivative of activation function * 
            # sum of all (local gradients * weights) of neurons associated with output of current layer, which is input to previous layer))
            
            #Initialize temp with containing all zeros
            temp = np.zeros(prev_weights.shape[0])

            #The loop iterates number of neurons in previous layer, as each neuron has one loc_grad
            for i in range(prev_loc_grads.size):
                #Compute local gradient times weights of the previous layer and add it to total
                #The multiplied weights is a vector
                temp += prev_loc_grads[i] * prev_weights[:, i]
            
            #Compute the  local gradient of hidden layer based on formula
            #local gradient of hidden layer = derivitive of sigmoid * sum of all (local grads * weights)
            self.loc_gradients = self.d_sigmoid() * temp
            
        #Update Weights and Biases, based on learning rate, local gradients and inputs to layer
        self.weights = self.weights + (learning_rate * np.outer(inputs_to_layer, self.loc_gradients))
        
        #The inputs_to_layer is ommited as bias is (conceptually) caused by input from a neuron with a fixed activation of 1
        self.biases = self.biases + learning_rate * self.loc_gradients

        #return updated weights and biases
        return self.weights, self.biases