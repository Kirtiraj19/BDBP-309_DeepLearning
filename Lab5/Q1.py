# XOR from scratch

import numpy as np


# -------------------- DATA --------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# Fix random values.
np.random.seed(42)


# -------------------- WEIGHTS --------------------

# Input layer:
# 2 inputs -> 4 hidden neurons.
# W1 is 2 x 4.
W1 = np.random.randn(2, 4)

# Four hidden biases.
b1 = np.zeros((1, 4))


# Hidden layer:
# 4 neurons -> 1 output neuron.
# W2 is 4 x 1.
W2 = np.random.randn(4, 1)

# One output bias.
b2 = np.zeros((1, 1))


# -------------------- SIGMOID --------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Learning rate.
lr = 0.5


# -------------------- TRAINING --------------------

for i in range(10000):

    # ===== FORWARD PASS =====

    # Hidden layer weighted sum.
    z1 = np.dot(X, W1) + b1

    # Hidden layer activation.
    a1 = sigmoid(z1)

    # Output weighted sum.
    z2 = np.dot(a1, W2) + b2

    # Final prediction.
    y_hat = sigmoid(z2)


    # ===== BACKWARD PASS =====

    # Error at output.
    error = y_hat - y

    # Gradient at output.
    d_z2 = error * y_hat * (1 - y_hat)

    # Gradient of second layer weights.
    d_W2 = np.dot(a1.T, d_z2)

    # Gradient of second layer bias.
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)


    # Send gradient back to hidden layer.
    d_a1 = np.dot(d_z2, W2.T)

    # Derivative of sigmoid at hidden layer.
    d_z1 = d_a1 * a1 * (1 - a1)

    # Gradient of first layer weights.
    d_W1 = np.dot(X.T, d_z1)

    # Gradient of first layer bias.
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)


    # ===== UPDATE WEIGHTS =====

    W1 = W1 - lr * d_W1
    b1 = b1 - lr * d_b1

    W2 = W2 - lr * d_W2
    b2 = b2 - lr * d_b2


# -------------------- TEST --------------------

# Do one final forward pass.
z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)

z2 = np.dot(a1, W2) + b2
y_hat = sigmoid(z2)


print("Predictions:")
print(y_hat)

print("\nRounded predictions:")
print(np.round(y_hat))
