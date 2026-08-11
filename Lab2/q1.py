#Lab 2 - Forward Pass
import numpy as np

# Q1: Initialize the network
# 4 input values
x = np.random.rand(4)

# Weights for each layer
W1 = np.random.rand(3, 4)   # 4 inputs 3 neurons
W2 = np.random.rand(2, 3)   # 3 neurons 2 neurons
W3 = np.random.rand(1, 2)   # 2 neurons 1 neuron

# Biases are 0
b1 = 0
b2 = 0
b3 = 0

# Q2 & Q3: Forward pass
# Hidden Layer 1
z1 = np.dot(W1, x) + b1
a1 = np.maximum(0, z1)      # ReLU
print("Hidden Layer 1")
print("Activation =", a1)

# Hidden Layer 2
z2 = np.dot(W2, a1) + b2
a2 = np.maximum(0, z2)      # ReLU
print("Hidden Layer 2")
print("Activation =", a2)

# Output Layer
z3 = np.dot(W3, a2) + b3
a3 = np.maximum(0, z3)      # ReLU

# Final prediction
y_hat = a3
print("Output Layer")
print("Activation =", a3)
print("Prediction =", y_hat)




# import numpy as np
# x = np.array([1, 2, 3, 4])  #input
# w = np.array([[0.4, 0.2, 0.3, 0.1]])  #weights
# b = np.array([0])  #bias
# #calculae weighted sum
# z = np.dot(w,x) #neuron calculates the dot product
# print("Weighted sum =",z)   # z = w11*x1 + w12*x2 + w13*x3 + w14*x4
#
# a = np.maximum(0, z)  #If z is positive, ReLU returns z and if negative ReLU returns 0.
# print("Activation value =",a)
#
# y_hat = a #since'a' is the output layer
# print("y_hat =",y_hat)
#
# #for second network
# x = np.array([1, 2, 3, 4]) #4 input neurons
# #weights from Input layer
# W1 = np.array([
#     [0.1, 0.2, 0.3, 0.4],   # weights for neuron 1
#     [0.5, 0.6, 0.7, 0.8],   # weights for neuron 2
#     [0.9, 1.0, 1.1, 1.2]    # weights for neuron 3
# ])
#
# # Bias for neurons
# b1 = 0
# z1 = np.dot(W1, x) + b1
# a1 = np.maximum(0, z1) #ReLu activation
# print("Hidden Layer 1:")
# print("z1:",z1)
# print("a1:",a1)
#
# #Hidden Layer 2
# W2 = np.array([
#     [0.1, 0.2, 0.3],    # weights for neuron 1
#     [0.2, 0.3, 0.4]     # weights for neuron 2
# ])
#
# b2 = 0
# z2 = np.dot(W2, a1) + b2
# a2 = np.maximum(0, z2)
# print("Hidden Layer 2:")
# print("z2:",z2)
# print("a2:",a2)
#
# #Output Layer
# W3 = np.array([    # weights for neuron 1
#     [0.1, 0.2]
# ])
# b3 = 0
# z3 = np.dot(W3, a2) + b3
# a3 = np.maximum(0, z3)
# y_hat = a3
# print("Output Layer:")
# print("z3:",z3)
# print("y_hat:",a3)


# Q1	Randomly initializes x, W1, W2, W3 and biases
# Q2	Performs forward pass and prints activations + ŷ
# Q3	Uses matrix/vector operations (np.dot) and no loops