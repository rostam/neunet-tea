import neunet, numpy as np

input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = neunet.NeuNetTee(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# several training
epochs = 5

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(",")
        correct_label = int(all_values[0])
        print(correct_label, "correct label ")
        inputs = (np.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets = np.zeros(output_nodes) + 0.01
        n.train(inputs, targets)

test_data_file =  open("mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []
for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    print(correct_label, "correct label ")
    inputs = (np.asfarray(all_values[1:])/255.0*0.99)+0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print(label, "network's answer")
    if label == correct_label:
        scores.append(1)
    else:
        scores.append(0)
    pass

print(scores)
scores = np.asarray(scores)
performance = scores.sum() / scores.size
print(performance)