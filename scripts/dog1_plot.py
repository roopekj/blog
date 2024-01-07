import matplotlib.pyplot as plt
import numpy as np
import imageio
from math import sqrt


def calc_loss(slope, x, y):
    return (y - f(slope, x))**2


def f(slope, x):
    return slope * x


def gradient_step(slope, x, y, learning_rate):
    gradient = 2*(y-slope*x)*(-x)
    return slope - learning_rate * gradient


lr = 0.9e-1
a = -4
X, Y = 3, 1

slope_range = np.arange(-5, 5, 0.1)
x_range = np.arange(-5, 5, 0.1)

fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=80)
ax[0].axhline(0, color='black', linewidth=.5)
ax[0].axvline(0, color='black', linewidth=.5)
ax[0].set_xlim([-5, 5])
ax[0].set_ylim([-5, 5])
ax[1].set_xlim([-5, 5])

# The plot of the line
line = [f(a, x) for x in x_range]
ax[0].scatter(X, Y)
ax[0].plot(x_range, line)
length = sqrt(1.0+a**2)
ax[0].arrow(x=0, y=0, dx=1/length, dy=a/length, width=5e-2)

# Loss function as a function of slope
losses = [calc_loss(slope, X, Y) for slope in slope_range]
ax[1].plot(slope_range, losses)

# Loss at starting state
loss = calc_loss(a, X, Y)
ax[1].scatter(a, loss)

fname = "./img_1_DOG/0.png"
fig.savefig("./img_1_DOG/0.png")

filenames = [fname]

# Gradient updates
for i in range(1, 20):
    ax[0].lines.pop(-1)

    new_a = gradient_step(a, X, Y, lr)
    loss = calc_loss(new_a, X, Y)
    ax[1].scatter(new_a, loss)

    line_x = np.linspace(a, new_a, num=10)
    line_y = np.linspace(calc_loss(a, X, Y), calc_loss(new_a, X, Y), num=10)
    ax[1].plot(line_x, line_y, linestyle="dotted")

    line = [f(new_a, x) for x in x_range]
    ax[0].plot(x_range, line)

    length = sqrt(1.0+new_a**2)
    ax[0].arrow(x=0, y=0, dx=1/length, dy=new_a/length, width=5e-2)

    fname = "./img_1_DOG/%s.png" % i
    fig.savefig(fname)
    filenames.append(fname)

    a = new_a

images = []
for filename in filenames:
    images.append(imageio.v2.imread(filename))
imageio.mimsave('../assets/gradient_1_dog.gif', images, "GIF", duration=1e3, loop=0)