import numpy as np
import mnist
import matplotlib.pyplot as plt
from matplotlib.image import imread
import skimage.measure
from scipy import signal
import math
from PIL import Image
# from resizeimage import resizeimage


def ReLU(x):
    return np.maximum(x, 0.0)


def softmax(x):
    # Change data type to compensate
    c = x.astype('float128')

    exp_x = np.exp(c)
    return exp_x / np.sum(exp_x)

def cross_entropy(y_pred, y_true):
    
    log = np.log(y_pred)
    log_loss = log * y_true
    return -np.sum(log_loss)

def L2_regularization(la, weight1, weight2):
    weight1_loss = 0.5 * la * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * la * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss

def format_data(train_images, train_labels, test_images, test_labels, num_classes):
    # data processing
    x_train = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
    x_train = x_train / 255 #normalization
    y_train = np.eye(num_classes)[train_labels] #convert label to one-hot

    x_test = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
    x_test = x_test / 255 #normalization
    y_test = test_labels

    return x_train, y_train, x_test, y_test

# Creates the Network Class
class Network:
    def __init__(self, num_nodes_in_layers, batch_size, num_epochs, learning_rate):

        # Assigns all of the parameter information about the network
        self.num_nodes_in_layers = num_nodes_in_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.accuracy_list = []
        self.iter_list = []
        self.training_iter = 0

        # Creates the weight and bias numpy arrays which will be used for the calculations
        self.weight1 = np.random.normal(0, 1, [self.num_nodes_in_layers[0], self.num_nodes_in_layers[1]])
        self.bias1 = np.zeros((1, self.num_nodes_in_layers[1]))
        self.weight2 = np.random.normal(0, 1, [self.num_nodes_in_layers[1], self.num_nodes_in_layers[2]])
        self.bias2 = np.zeros((1, self.num_nodes_in_layers[2]))
        self.loss = []

    def get_accuraccy_while_training(self):
        accuracy = self.test(x_test, y_test)
        self.accuracy_list.append(accuracy)
        self.iter_list.append(self.training_iter)
        self.training_iter += 1

    # Forward propogation function X > W1 > W2 > Y
    def forward(self, x):

        z1 = np.dot(x, self.weight1) + self.bias1
        self.a1 = ReLU(z1)

        z2 = np.dot(self.a1, self.weight2) + self.bias2
        output = softmax(z2)

        return output        
    
    def loss_and_backprop(self, y, inputs_batch, labels_batch):
        
        loss = cross_entropy(y, labels_batch)
        

        # Compute the derivative of loss (d/dx of MSE function)
        delta_y = 2 * (y - labels_batch)
        delta_hidden_layer = np.dot(delta_y, self.weight2.T) 
        
        # d/dx of the ReLU func
        delta_hidden_layer[self.a1 <= 0] = 0

        # Backpropogate the error to the layer 2 and 1
        weight2_gradient = np.dot(self.a1.T, delta_y) # forward * backward
        bias2_gradient = np.sum(delta_y, axis = 0, keepdims = True)
    
        weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer)
        bias1_gradient = np.sum(delta_hidden_layer, axis = 0, keepdims = True)

        # L2 regularization
        weight2_gradient += 0.01 * self.weight2
        weight1_gradient += 0.01 * self.weight1

        # SGD, update the weights w/ the learning rate
        self.weight1 -= self.learning_rate * weight1_gradient 
        self.bias1 -= self.learning_rate * bias1_gradient
        self.weight2 -= self.learning_rate * weight2_gradient
        self.bias2 -= self.learning_rate * bias2_gradient

        # self.get_accuraccy_while_training()

        print('=== Loss: {:.2f} ==='.format(loss))

    def test(self, inputs, labels):

        input_layer = np.dot(inputs, self.weight1) 
        hidden_layer = ReLU(input_layer + self.bias1)
        scores = np.dot(hidden_layer, self.weight2) + self.bias2
        probs = softmax(scores)
        print(scores[0])
        print(probs[0])
        acc = float(np.sum(np.argmax(probs, 1) == labels)) / float(len(labels))
        print('Test accuracy:', acc*100)
        return acc*100

    
    # Train the network give then input data and realized output values
    def train(self, inputs, labels):
        for epoch in range(self.num_epochs): # training begin
            iteration = 0
            while iteration < len(inputs):

                # The batch data and batch labels
                inputs_batch = inputs[iteration:iteration+self.batch_size]
                labels_batch = labels[iteration:iteration+self.batch_size]

                # Forward and back propogation for the current sample
                y = self.forward(inputs_batch)
                self.loss_and_backprop(y,inputs_batch, labels_batch)

                iteration += self.batch_size
            

net = Network(
                 num_nodes_in_layers = [784, 50, 10], 
                 batch_size = 1,
                 num_epochs = 5,
                 learning_rate = 0.001, 
             )

# Load data from gzip files and train the network
num_classes = 10
train_images = mnist.train_images() 
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

x_train, y_train, x_test, y_test = format_data(train_images, 
                                                train_labels, 
                                                test_images, 
                                                test_labels, 
                                                num_classes)

net.train(x_train, y_train)

print("Testing...")
net.test(x_test, y_test)

plt.plot(net.iter_list, net.accuracy_list)
plt.show()

