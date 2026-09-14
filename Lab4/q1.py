# Full Neural Network Implementation from Scratch

import numpy as np
import matplotlib.pyplot as plt


# -------------------- DATA --------------------

# Input data.
# Each row is one training example.
X = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
], dtype=float)

# Correct answers.
y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# -------------------- INITIALIZATION --------------------

# Fix the random values so results are repeatable.
np.random.seed(42)

# Three input values -> one output neuron.
# Therefore W has shape 3 x 1.
W = np.random.randn(3, 1)

# One bias for the output neuron.
b = np.random.randn(1)


# -------------------- SIGMOID --------------------

def sigmoid(x):
    # Sigmoid converts a value into a value between 0 and 1.
    return 1 / (1 + np.exp(-x))


# Learning rate controls how big each weight update is.
lr = 0.1

# Store loss from every iteration.
loss_list = []


# -------------------- TRAINING --------------------

for i in range(1000):

    # ===== FORWARD PASS =====

    # Calculate weighted sum.
    z = np.dot(X, W) + b

    # Convert weighted sum into prediction.
    y_hat = sigmoid(z)


    # ===== LOSS =====

    # Calculate mean squared error.
    loss = np.mean((y_hat - y)**2)

    # Save the loss for plotting later.
    loss_list.append(loss)


    # ===== BACKWARD PASS =====

    # Error between prediction and actual value.
    error = y_hat - y

    # Derivative of sigmoid.
    sigmoid_d = y_hat * (1 - y_hat)

    # Gradient with respect to z.
    d_z = error * sigmoid_d

    # Gradient of weights.
    d_W = np.dot(X.T, d_z)

    # Gradient of bias.
    d_b = np.sum(d_z)


    # ===== UPDATE =====

    # Move weights in the direction that reduces loss.
    W = W - lr * d_W

    # Update bias.
    b = b - lr * d_b


# -------------------- RESULT --------------------

print("Final predictions:")
print(y_hat)

# Round values to 0 or 1.
print("\nRounded predictions:")
print(np.round(y_hat))

# -------------------- LOSS PLOT --------------------
plt.plot(loss_list)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.show()
