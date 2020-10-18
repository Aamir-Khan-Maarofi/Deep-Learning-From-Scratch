#Importing numpy for vector operations
import numpy as np

class Dense():
    """ 
    Dense represent a layer in fully connected multilayer perceptron network

    ...

    Attributes
    ----------
    weights : 2D ndarray, shape = (n_inputs, n_nodes)
        Stores the weights of all neurons in the layer
    outputs : 2D ndarray of 'ones', shape = (n_nodes, 1)
        Stores the outputs of all neurons in the layer
    biases : 2D ndarray of 'ones', shape = (n_nodes, 1)
        Stores the biases of all neurons in the layer
    activation_potentials : 2D ndarray 'zeros', shape = (n_nodes, 1)
        Store the  activation_potentials of all the neurons in the layer
    local_grads : 2D ndarray 'zeros', shape = (n_nodes, 1)
        Store local gradients of all neurons in the layer
    
    Methods 
    -------
    Constructor(n_inputs, n_nodes) :
        Initialize all the attributes mentioned above, as  
    act_poten() : 
        Calculate activation potential of the current layer neurons
    sigmoid(inputs) : 
        Calculate activation potential of the current layer neurons
        Using formula inputs * weights + biases
    d_sigmoid() :
        Activates the activation_potential -> ndarray (act_potentials of all neurons in layer)
        Save the activated ndarry to outputs -> ndarray (outputs of all neurons in this layer)
    loc_grad(acc_error, layer, network) : 

    updater() : 

    forward_pass() : 

    backward_pass() : 

    """
    def __init__(self, n_inputs, n_nodes):
        #Weights associated with all n_nodes for n_inputs synaptic inputs
        #Each column in the weight matrix is weight vector associated with one neuron
        self.weights = np.random.uniform(low=0, high=1, size=n_inputs*n_nodes).reshape(n_inputs, n_nodes)

        #Biases associated to each neuron, shape (1, n_nodes)
        #There are n_nodes cloumns and each one represent bias associated with one neuron in layer
        #This one dimension array will be added to linear cobinations of all neurons  
        #assuming the synaptic connection associated to biases is '1'
        self.biases = np.ones(n_nodes)
                
    def act_poten(self, inputs):
        #Calculate activation potential of the current layer neurons -> ndarray 
        #Which is inputs times weights add a bias
        #Stores it to activation_potentials
        self.activation_potentials = np.dot(inputs.T, self.weights) + self.biases
    
    def sigmoid(self):
        #Activates the activation_potential -> ndarray
        #Save the activated ndarry to outputs
        self.outputs = 1 / (1 + np.exp(-self.activation_potentials))

    def d_sigmoid(self):
        #Derivitive of the activation potential -> ndarray (For all neurons, values of first differentail activation function at activation_potential)
        #Will be used in local gradient calculation
        return np.dot(self.outputs, (1 - self.outputs))

    def loc_grad(self, targets, prev_loc_grads, prev_weights, layer, network):
        #Calculation of local gradients of all neurons in this layer -> ndarray
        #Save this to local gradietns -> ndarray
        if layer == network[-1]:
            #Output layer = error_at_end * derivative of activation function
            error_at_end = (1/2) * np.power((layer.outputs - targets), 2)
            self.loc_gradients = np.dot(error_at_end, self.d_sigmoid())
        else:
            # Hidden layers => derivative of activation function * sum of all (derivative of activation
            # function of previous layer * (loc_grad vector of previous layer * weights vector of previous layer
            # associated with output of current layer, which is input to previous layer))
            self.loc_gradients = np.dot(self.d_sigmoid(), np.dot(prev_loc_grads, prev_weights))

    def updater(self):
        #Update Weights and Biases, based on learning rate, local gradients and error
        
        pass

    def forward_pass(self, inputs):
        #inputs times weights add a bias and activate
        #calculate error,  this doesn't fit here, but will find it out
        self.act_poten(inputs)
        self.sigmoid()

    def backward_pass(self, targets, prev_loc_grads, prev_weights, layer, network):
        #Calculate local gradient of all the neurons in this layer
        #Update weights and biases
        self.loc_grad(targets, prev_loc_grads, prev_weights, layer, network)
        print('Got (targets)        : ', targets)
        print('Shape                : ', targets.shape)
        print('Got (prev_loc_grad)  : ', prev_loc_grads)
        print('Shape                : ', prev_loc_grads.shape)
        print('Got (prev_weights)   : ', prev_weights)
        print('Shape                : ', prev_weights.shape)
        print('Got (layer)          : ', layer)
        print('Got (network)        : ', network)
        print('Local Gradients: ', self.loc_gradients)
        print('Shape                : ', self.loc_gradients.shape)

