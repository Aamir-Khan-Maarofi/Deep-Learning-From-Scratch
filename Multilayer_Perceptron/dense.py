#Importing numpy for vector operations
import numpy as np

class Dense():
    """ 
    Dense represent a layer in fully connected multilayer perceptron network

    ...

    Attributes
    ----------
    weights : 2D ndarray, shape = (input_dim, num_nodes)
        Stores the weights of all neurons in the layer
    outputs : 2D ndarray of 'ones', shape = (num_nodes, 1)
        Stores the outputs of all neurons in the layer
    biases : 2D ndarray of 'ones', shape = (num_nodes, 1)
        Stores the biases of all neurons in the layer
    activation_potentials : 2D ndarray 'zeros', shape = (num_nodes, 1)
        Store the  activation_potentials of all the neurons in the layer
    local_grads : 2D ndarray 'zeros', shape = (num_nodes, 1)
        Store local gradients of all neurons in the layer
    
    Methods 
    -------
    Constructor(input_dim, num_nodes) :
        Initialize all the attributes mentioned above, as  
    act_poten() : 
        Calculate activation potential of the current layer neurons
    activate(inputs) : 
        Calculate activation potential of the current layer neurons
        Using formula inputs * weights + biases
    d_activate() :
        Activates the activation_potential -> ndarray (act_potentials of all neurons in layer)
        Save the activated ndarry to outputs -> ndarray (outputs of all neurons in this layer)
    loc_grad(acc_error, layer, network) : 

    updater() : 

    forward_pass() : 

    backward_pass() : 

    """
    def __init__(self, input_dim, num_nodes):
        #Weights associated with all num_nodes for input_dim synaptic inputs
        self.weights = np.random.uniform(low=0, high=1, size=(input_dim, num_nodes))

        #Outputs of all the neurons in the layer, shape (num_nodes, 1)
        self.outputs = np.ones((num_nodes, 1))

        #Biases associated to each neuron, assuming the synaptic connection associated to biases is '1'
        self.biases = np.ones((num_nodes, 1))
        
        #activation_potentials (Linear Combination) of all neurons in the layer
        self.activation_potentials = np.zeros((num_nodes, 1))
        
        #Local Gradients associated with each node of the current layer
        self.loacal_grads = np.zeros((num_nodes, 1))
    
    def act_poten(self, inputs):
        #Calculate activation potential of the current layer neurons -> ndarray 
        #Formula (Input * Weight) + bias 
        self.activation_potentials = np.dot(self.weights, inputs.T) + self.biases
    
    def activate(self):
        #Activates the activation_potential -> ndarray (act_potentials of all neurons in layer)
        #Save the activated ndarry to outputs -> ndarray (outputs of all neurons in this layer)
        self.outputs = 1 / (1 - np.exp(self.activation_potentials))

    def d_activate(self):
        #Derivitive of the activation potential -> ndarray (For all neurons, values of first differentail activation function at activation_potential)
        #Will be used in local gradient calculation
        return self.activation_potentials * (1 - self.activation_potentials)

    def loc_grad(self, acc_error, layer, network):
        #Calculation of local gradients of all neurons in this layer -> ndarray
        #Save this to local gradietns -> ndarray
        if layer == network[-1]:
            # Output layer = error * derivative of activation function
            
            pass
        else:
            # Hidden layers = derivative of activation function * sum of all (derivative of activation
            # function of next layer * weights associated with this neuron going to all neurons in next layer)
            pass

    def updater(self):
        #Update Weights and Biases, based on learning rate, local gradients and error
        
        pass

    def forward_pass(self):
        #inputs times weights add a bias and activate
        #calculate error,  this doesn't fit here, but will find it out
        pass

    def backward_pass(self):
        #Calculate local gradient of all the neurons in this layer
        #Update weights and biases
        pass