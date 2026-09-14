import torch
import torch.nn as nn


# =========================================================
# 1. BATCH NORMALIZATION
# =========================================================

# This model has 10 input values and 20 neurons.
model_batch = nn.Sequential(

    nn.Linear(10, 20),

    # BatchNorm1d normalizes the output of the linear layer.
    nn.BatchNorm1d(20),

    nn.ReLU(),

    nn.Linear(20, 2)
)

print("Batch Normalization model:")
print(model_batch)


# =========================================================
# 2. LAYER NORMALIZATION
# =========================================================

model_layer = nn.Sequential(

    nn.Linear(10, 20),

    # LayerNorm normalizes the features of each sample.
    nn.LayerNorm(20),

    nn.ReLU(),

    nn.Linear(20, 2)
)

print("\nLayer Normalization model:")
print(model_layer)


# =========================================================
# 3. DROPOUT
# =========================================================

model_dropout = nn.Sequential(

    nn.Linear(10, 20),

    nn.ReLU(),

    # During training, approximately 50% of neurons
    # are randomly turned off.
    # This helps reduce overfitting.
    nn.Dropout(0.5),

    nn.Linear(20, 2)
)

print("\nDropout model:")
print(model_dropout)


# =========================================================
# 4. SGD
# =========================================================

# SGD means Stochastic Gradient Descent.
optimizer_sgd = torch.optim.SGD(
    model_dropout.parameters(),
    lr=0.01
)

print("\nSGD optimizer:")
print(optimizer_sgd)


# =========================================================
# 5. SGD WITH MOMENTUM
# =========================================================
# Momentum uses information from previous updates
# to make learning smoother.
optimizer_momentum = torch.optim.SGD(
    model_dropout.parameters(),
    lr=0.01,
    momentum=0.9
)

print("\nSGD with Momentum:")
print(optimizer_momentum)
