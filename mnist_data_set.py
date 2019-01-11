import neunet, numpy as np, matplotlib.pyplot as plt

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = neunet.NeuNetTee(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(",")
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

test_data_file = open("mnist_test_10.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

all_values = test_data_list[0].split(',')
print(all_values[0])

image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap="Greys", interpolation='None')
plt.show()
res = n.query((np.asfarray(all_values[1:])/255.0*0.99) + 0.01)
print(res)
