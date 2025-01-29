import random

import data_operations
import numpy as np
import NeuralNetwork

(data_y, data_x) = data_operations.read_data("mnist_train.csv")
(test_y, test_x) = data_operations.read_data("mnist_test.csv")

num_classes = 10
y_train_onehot = np.eye(num_classes)[data_y]
y_test_onehot = np.eye(num_classes)[test_y]

input_size = data_x.shape[1]
hidden_size = 32
output_size = num_classes
learning_rate = 0.8
epochs = 100

model = NeuralNetwork.NeuralNetwork(input_size, hidden_size, output_size)
model.train(data_x, y_train_onehot, learning_rate, epochs, test_x, test_y)

while True:
    number = int(input("Enter number 0-9 : "))
    if not -1 < number < 10:
        break
    indexes = [i for i, x in enumerate(test_y) if x == number]
    index = random.choice(indexes)
    print(f"Random {number} from test data")
    vector = test_x.iloc[index]
    data_operations.show_image(vector)
    pred = model.predict(vector)
    print(f"Prediction : {pred}")
