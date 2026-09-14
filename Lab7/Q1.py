# This program creates a simple MNIST classifier.
#
# MNIST contains 28 x 28 grayscale images of digits 0-9.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -------------------- DATA --------------------

# ToTensor converts images into PyTorch tensors.
transform = transforms.ToTensor()


# Download/load training data.
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)


# Download/load testing data.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# Create batches for training.
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)


# Create batches for testing.
test_loader = DataLoader(
    test_data,
    batch_size=64
)


# -------------------- MODEL --------------------

class Model(nn.Module):

    def __init__(self):

        # Initialize the parent PyTorch class.
        super().__init__()

        # Sequential means these layers are used one after another.
        self.net = nn.Sequential(

            # Convert 28 x 28 image into one long vector.
            nn.Flatten(),

            # 784 inputs -> 128 hidden neurons.
            nn.Linear(28 * 28, 128),

            # Add non-linearity.
            nn.ReLU(),

            # 128 hidden neurons -> 10 output neurons.
            # One output for each digit 0-9.
            nn.Linear(128, 10)
        )


    def forward(self, x):

        # Send input through the network.
        return self.net(x)


# Create the model.
model = Model()


# -------------------- LOSS --------------------

# CrossEntropyLoss is commonly used for classification.
loss_function = nn.CrossEntropyLoss()


# -------------------- OPTIMIZER --------------------

# SGD updates the model weights.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# -------------------- TRAINING --------------------

for epoch in range(5):

    # Go through all training batches.
    for X, y in train_loader:

        # Forward pass.
        prediction = model(X)

        # Calculate loss.
        loss = loss_function(prediction, y)

        # Remove gradients from previous iteration.
        optimizer.zero_grad()

        # Calculate new gradients.
        loss.backward()

        # Update weights.
        optimizer.step()


    print(
        "Epoch:",
        epoch + 1,
        "Loss:",
        loss.item()
    )
