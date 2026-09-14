# Vanishing:
# gradient becomes smaller and approaches zero.
#
# Exploding:
# gradient becomes larger and larger.

import matplotlib.pyplot as plt


# Number of layers.
layers = 30


# =========================================================
# VANISHING GRADIENT
# =========================================================

# Start with gradient = 1.
gradient = 1.0

# Store the gradient after every layer.
vanish = []

for i in range(layers):

    # Multiplying by 0.5 makes the value smaller
    # at every layer.
    gradient = gradient * 0.5

    # Save the value.
    vanish.append(gradient)


# =========================================================
# EXPLODING GRADIENT
# =========================================================

# Start again with gradient = 1.
gradient = 1.0

# Store exploding values.
explode = []

for i in range(layers):

    # Multiplying by 1.5 makes the value larger
    # at every layer.
    gradient = gradient * 1.5

    # Save the value.
    explode.append(gradient)


# =========================================================
# PLOT VANISHING GRADIENT
# =========================================================

plt.figure()

plt.plot(
    range(1, layers + 1),
    vanish
)

plt.xlabel("Layer")
plt.ylabel("Gradient")
plt.title("Vanishing Gradient")

plt.show()


# =========================================================
# PLOT EXPLODING GRADIENT
# =========================================================

plt.figure()

plt.plot(
    range(1, layers + 1),
    explode
)

plt.xlabel("Layer")
plt.ylabel("Gradient")
plt.title("Exploding Gradient")

plt.show()
