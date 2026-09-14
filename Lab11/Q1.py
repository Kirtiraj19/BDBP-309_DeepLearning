# CNN from scratch and maxpooling

import numpy as np
# =========================================================
# 1. CONVOLUTION
# =========================================================

# Create a random 32 x 32 grayscale image.
image = np.random.rand(32, 32)


# Create a 3 x 3 kernel.
# A kernel is a small matrix used to detect features.
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])


# With no padding and stride 1:
#
# output size = input size - kernel size + 1
#             = 32 - 3 + 1
#             = 30
#
# Therefore output is 30 x 30.
output = np.zeros((30, 30))


# Move the kernel over the image.
for i in range(30):

    for j in range(30):

        # Take a 3 x 3 part of the image.
        part = image[
            i:i+3,
            j:j+3
        ]

        # Multiply the image part by the kernel.
        # Then add all values.
        output[i, j] = np.sum(
            part * kernel
        )


print("Image shape:")
print(image.shape)

print("\nKernel shape:")
print(kernel.shape)

print("\nConvolution output shape:")
print(output.shape)


# =========================================================
# 2. MAX POOLING
# =========================================================

# Small example image.
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])


# We use a 2 x 2 pooling window.
pool = 2


# 4 x 4 input with 2 x 2 pool gives 2 x 2 output.
output = np.zeros((2, 2))


# Move the pooling window.
for i in range(2):

    for j in range(2):

        # Take one 2 x 2 region.
        part = image[
            i*2:i*2+2,
            j*2:j*2+2
        ]

        # Keep only the largest value.
        output[i, j] = np.max(part)


print("\nOriginal image:")
print(image)

print("\nAfter max pooling:")
print(output)
