import numpy as np
import scipy.special as scs


class NeuNetTee:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input = input_nodes
        self.hidden = hidden_nodes
        self.output = output_nodes
        self.lr = learning_rate

        self.weights_input_hidden = np.random.normal(0.0, pow(self.hidden, -0.5), (self.hidden, self.input))
        self.weights_hidden_output = np.random.normal(0.0, pow(self.output, -0.5), (self.output, self.hidden))

        self.activate_function = lambda x: scs.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activate_function(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activate_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        self.weights_hidden_output += self.lr * np.dot((output_errors*final_outputs*(1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.weights_input_hidden += self.lr * np.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activate_function(hidden_inputs)
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activate_function(final_inputs)
        return final_outputs
