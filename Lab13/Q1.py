# A simple representation for DNA is one-hot encoding:

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# 1. ONE-HOT ENCODING
# =========================================================

# Example DNA sequence.
sequence = "ACGT"


# Dictionary tells us how each DNA base is represented.
encode = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1]
}


# Empty list to store encoded bases.
data = []


# Look at each base in the DNA sequence.
for base in sequence:

    # Get the four numbers for that base.
    data.append(encode[base])


# Convert the list into a NumPy array.
data = np.array(data)


print("DNA sequence:")
print(sequence)

print("\nOne-hot encoded sequence:")
print(data)


# =========================================================
# 2. CNN MODEL
# =========================================================
#
# For a real dataset, the DNA sequences must first be
# converted into tensors with the correct shape.
#
# Conv1d expects:
# batch x channels x sequence_length
#
# Here the four channels represent A, C, G and T.

class DNA_CNN(nn.Module):

    def __init__(self, length):

        super().__init__()

        # First convolution.
        # Input channels = 4 because we have A,C,G,T.
        self.conv1 = nn.Conv1d(
            4,
            32,
            kernel_size=5
        )

        # Second convolution.
        self.conv2 = nn.Conv1d(
            32,
            64,
            kernel_size=5
        )

        # Max pooling reduces sequence size.
        self.pool = nn.MaxPool1d(2)

        # Calculate the sequence length after:
        # first convolution: length - 5 + 1
        # pooling: divide by 2
        # second convolution: - 5 + 1
        # pooling: divide by 2
        #
        # This calculation is used to know the input size
        # of the final linear layer.
        length = (length - 5 + 1) // 2
        length = (length - 5 + 1) // 2

        # Final layer gives one value for gene accessibility.
        self.fc = nn.Linear(
            64 * length,
            1
        )


    def forward(self, x):

        # First convolution.
        x = self.conv1(x)

        # ReLU.
        x = torch.relu(x)

        # Pooling.
        x = self.pool(x)

        # Second convolution.
        x = self.conv2(x)

        # ReLU.
        x = torch.relu(x)

        # Pooling.
        x = self.pool(x)

        # Convert to one long vector.
        x = x.flatten(1)

        # Final prediction.
        x = self.fc(x)

        return x


# Example sequence length.
# Change this to the length used in your actual dataset.
length = 100

model = DNA_CNN(length)

print("\nCNN model:")
print(model)
