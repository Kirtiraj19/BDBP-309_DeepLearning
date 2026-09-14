# Backward pass - Backpropagation

import numpy as np

# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

x = np.array([1.0, 2.0, 3.0, 4.0])

# Actual answer
y = 1.0


# ---------------------------------------------------------
# Network:
#
# 4 inputs
#    ↓
# 3 hidden neurons
#    ↓
# 1 output neuron
# ---------------------------------------------------------

W1 = np.random.randn(3, 4)
b1 = np.random.randn(3)

W2 = np.random.randn(1, 3)
b2 = np.random.randn(1)


# =========================================================
# FORWARD PASS
# =========================================================

z1 = np.dot(W1, x) + b1
a1 = np.maximum(0, z1)

z2 = np.dot(W2, a1) + b2
y_hat = np.maximum(0, z2)

print("Prediction:", y_hat)


# =========================================================
# LOSS
# =========================================================

loss = 0.5 * (y_hat - y)**2

print("Loss:", loss)


# =========================================================
# BACKWARD PASS
# =========================================================

# ---------------------------------------------------------
# Gradient of loss with respect to prediction
#
# dL/dy_hat = y_hat - y
# ---------------------------------------------------------

dL_dy = y_hat - y


# ---------------------------------------------------------
# Derivative of ReLU at output
# ---------------------------------------------------------

drelu_output = (z2 > 0).astype(float)

# Gradient with respect to z2
dL_dz2 = dL_dy * drelu_output


# ---------------------------------------------------------
# Gradient of W2
#
# dL/dW2 = dL/dz2 * a1
# ---------------------------------------------------------

dW2 = np.outer(dL_dz2, a1)

# Gradient of bias
db2 = dL_dz2


# ---------------------------------------------------------
# Gradient flowing back to hidden layer
# ---------------------------------------------------------

dL_da1 = np.dot(W2.T, dL_dz2)


# ---------------------------------------------------------
# ReLU derivative for hidden layer
# ---------------------------------------------------------

drelu_hidden = (z1 > 0).astype(float)

dL_dz1 = dL_da1 * drelu_hidden


# ---------------------------------------------------------
# Gradient of first layer weights
# ---------------------------------------------------------

dW1 = np.outer(dL_dz1, x)

# Gradient of first layer bias
db1 = dL_dz1


# ---------------------------------------------------------
# Print gradients
# ---------------------------------------------------------

print("Gradients:")
print("\ndW1:")
print(dW1)
print("\ndb1:")
print(db1)
print("\ndW2:")
print(dW2)
print("\ndb2:")
print(db2)