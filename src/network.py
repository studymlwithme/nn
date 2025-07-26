from src.enums import activations
from src.enums import losses
from src.data_formats import ActivatedLayer
from src.data_formats import LossLayer
from src.data_formats import OutputLayer
import numpy as np
from src.mapping import LAYER_MAPPING
from src.mapping import LOSS_MAPPING
from src.mapping import OUTPUT_LAYER_MAPPING

class Network:
    def __init__(self, learning_rate: float = .001):
        self.learning_rate = learning_rate
        self.next_layer = None
        self.layer_head = None
        self.layer_tail = None
        self.predictions = None

    def add_input_layer(self, inputs: np.ndarray):
        if self.layer_head is None and self.next_layer is None:
            self.next_layer = ActivatedLayer(neurons=inputs, features_in=None, features_out=inputs.shape[1], weights=None, biases=None)
            self.layer_head = ActivatedLayer(None, None, None, None, None, None, None, self.next_layer)
            self.layer_head.next = self.next_layer
            self.next_layer.prev = self.layer_head
        else:
            raise ValueError("You must add the input layer before adding any hidden layers.")
        
    def add_output_layer(self, features_in: int, features_out: int, y_true: np.ndarray, weighted = True, output_activation: activations = 'softmax'):
        init_weights = None
        init_biases = None 
        if weighted:
            std = np.sqrt(2 / features_in)            
            init_weights = np.random.normal(0, std, (features_in, features_out))
            init_biases = np.zeros(features_out)
        next_layer = OutputLayer(features_in, features_out, y_true, init_weights, init_biases, output_activation)
        if self.layer_head is None and self.next_layer is None:
            raise ValueError("You must add the input layer using .add_input_layer() before adding any hidden layers.")
        else:
            self.next_layer.next = next_layer
            self.next_layer.next.prev = self.next_layer
            self.next_layer = self.next_layer.next

    def add_loss(self, loss_type: losses, y_true: np.ndarray):
        next_layer = LossLayer(self.next_layer.features_out, y_true, loss_type)
        if self.layer_head is None and self.next_layer is None:
            raise ValueError("You must add the input layer using .add_input_layer() before adding any hidden layers.")
        else:
            self.next_layer.next = next_layer
            self.next_layer.next.prev = self.next_layer

    def add_hidden_layer(self, features_in: int, features_out: int,  activation: activations = "relu"):
        std = np.sqrt(2 / features_in)    
        init_weights = np.random.normal(0, std, (features_in, features_out))
        init_biases = np.zeros(features_out)
        next_layer = ActivatedLayer(features_in, features_out, init_weights, init_biases, activation)
        if self.layer_head is None and self.next_layer is None:
            raise ValueError("You must add the input layer using .add_input_layer() before adding any hidden layers.")
        else:
            self.next_layer.next = next_layer
            self.next_layer.next.prev = self.next_layer
            self.next_layer = self.next_layer.next

    def forward(self, X_test: np.ndarray = None, Y_test: np.ndarray = None) -> np.ndarray:
        layer = self.layer_head.next
        if X_test is not None and Y_test is not None:
            layer.neurons = X_test
        while True:
            if layer.next == None:
                self.layer_tail = layer
                return layer.neurons
            if not isinstance(layer.next, LossLayer):
                if isinstance(layer.next, OutputLayer): 
                    if layer.next.weights is None:
                        z =  layer.neurons
                    else:
                        z = layer.neurons @ layer.next.weights + layer.next.biases
                    if X_test is not None and Y_test is not None:
                        layer.next.y_true = Y_test
                    neurons, derivative = OUTPUT_LAYER_MAPPING[layer.next.activation](z, layer.next.y_true)
                elif layer.neurons.shape[1] != layer.next.weights.shape[0]:
                    raise ValueError("The input neurons for the L+1 layer were not set to the same dimension as its inputs.")
                else:
                    z = layer.neurons @ layer.next.weights + layer.next.biases
                    neurons, derivative = LAYER_MAPPING[layer.next.activation](z)
            else:
                z =  layer.neurons
                self.predictions = layer.neurons.argmax(axis=1)
                if X_test is not None and Y_test is not None:
                    layer.next.y_true = Y_test
                neurons, derivative = LOSS_MAPPING[layer.next.loss_type](z, layer.next.y_true)
            layer.next.neurons = neurons
            layer.next.derivative = derivative
            layer = layer.next


    def backward(self):
        start = True
        curr_error = None
        curr_layer = self.layer_tail
        while curr_layer.prev:
            
            if start:
                if not isinstance(curr_layer, LossLayer):
                    raise ValueError("You must define a loss layer in order to run back-propogation.")
                curr_error = curr_layer.derivative * curr_layer.prev.derivative
                start = False
                curr_layer = curr_layer.prev
                continue
            if curr_layer.biases is not None: 
                bias_jac = curr_error * 1
                curr_layer.biases -= self.learning_rate * np.average(bias_jac, axis=0)
            if curr_layer.weights is not None:
                weight_jac =  curr_layer.prev.neurons.T @ curr_error / curr_error.shape[0]# TODO ensure this is being broadcast across the correct axis
                print(f"Current Error: {curr_error} | Current Layer Activations: {curr_layer.prev.neurons.shape} Current Layer Weights: {curr_layer.weights.shape}")
                
                curr_error =  curr_error @  curr_layer.weights.T
                curr_layer.weights -= self.learning_rate * weight_jac
            if curr_layer.prev.derivative is not None:
                curr_error = curr_error * curr_layer.prev.derivative
            curr_layer = curr_layer.prev


    def mini_batch_sgd(self):
        pass

    


            
        
