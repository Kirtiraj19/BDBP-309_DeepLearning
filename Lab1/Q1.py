
import numpy as np
import matplotlib.pyplot as plt
# Generate 100 equally spaced values between -10 and 10
z = np.linspace(-10, 10, 100)

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Compute values
y = sigmoid(z)
dy = sigmoid_derivative(z)

# Plot
plt.figure(figsize=(8,5))
plt.plot(z, y, label="Sigmoid")
plt.plot(z, dy, label="Derivative")
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()


# Tanh Function
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Derivative
def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

# Compute values
y = tanh(z)
dy = tanh_derivative(z)

# Plot
plt.figure(figsize=(8,5))
plt.plot(z, y, label="Tanh")
plt.plot(z, dy, label="Derivative")
plt.title("Tanh Function")
plt.xlabel("z")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# ReLU Function
def relu(x):
    return np.maximum(0, x)

# Derivative
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Compute values
y = relu(z)
dy = relu_derivative(z)

# Plot
plt.figure(figsize=(8,5))
plt.plot(z, y, label="ReLU")
plt.plot(z, dy, label="Derivative")
plt.title("ReLU Function")
plt.xlabel("z")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# Leaky ReLU Function
def leaky_relu(x):
    return np.where(x > 0, x, 0.01*x)

# Derivative
def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

# Compute values
y = leaky_relu(z)
dy = leaky_relu_derivative(z)

# Plot
plt.figure(figsize=(8,5))
plt.plot(z, y, label="Leaky ReLU")
plt.plot(z, dy, label="Derivative")
plt.title("Leaky ReLU Function")
plt.xlabel("z")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# Softmax Function
def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)

# Output
softmax_output = softmax(z)

print("Softmax Output:")
print(softmax_output)


#OBSERVATIONS:

#The sigmoid function produces an S-shaped curve and the output lies between 0 and 1.
#The output is not zero-centred because it always produces positive values.
#For very large positive or negative input values, the curve becomes almost flat and the gradient approaches 0.
#The gradient is maximum around the input value 0 and is approximately 0.25.
#-------------
#The tanh function also produces an S-shaped curve and the output lies between −1 and +1.
#The output is zero-centred, and the function passes through (0, 0).
#For very large positive or negative input values, the gradient approaches 0,and gradient is maximum at z = 0, where its value is 1.
#-------------
#ReLU outputs 0 for all negative input values and increases linearly for positive input values
#and the minimum value is 0 & maximum value is infinity.
#The output is not zero-centred because it always produces positive values.
#For negative input values, the gradient is 0, so the neuron does not learn in that region. This may lead to the dying ReLU problem.
#For positive input values, the gradient remains 1
#-------------
#Leaky ReLU gives -infinity for minimum value and +infinity for maximum value.
#The output is not zero-centred.
#For negative input values, the gradient is a small constant, usually 0.01, instead of 0.
#For positive input values, the gradient is 1. And Leaky ReLU reduces the dying ReLU problem and allows learning to continue even for negative inputs.
#-------------
#Softmax converts a set of input values into probability and output value lies between 0 and 1.
#The outputs are not zero-centred. And The sum of all Softmax outputs is always 1.
#For very large positive input values, the corresponding probability approaches 1, while the probabilities of other classes approach 0.

#-----------------
#Relationship Between Sigmoid and Tanh
#Sigmoid and tanh are closely related S-shaped activation functions. The sigmoid function maps values to the range 0 to 1,
#whereas tanh maps values to −1 to +1 and is zero-centred.
#Tanh = 2*Sigmoid - 1