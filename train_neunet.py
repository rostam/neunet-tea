import neunet

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3

n = neunet.NeuNetTee(input_nodes, hidden_nodes, output_nodes, learning_rate)
res = n.query([1.0, 0.5, -1.5])
print(res)