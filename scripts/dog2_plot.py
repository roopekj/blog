import matplotlib.pyplot as plt
import numpy as np
import imageio
from math import sqrt
from matplotlib import cm


def calc_loss(slope, offset, x, y):
    return (y - f(slope, offset, x))**2


def f(slope, offset, x):
    return slope * x + offset


def gradient_step(slope, offset, x, y, learning_rate):
    slope_gradient = 2*(y-slope*x-offset)*(-x)
    offset_gradient = 2*(y-slope*x-offset)*(-1)
    return slope - learning_rate * slope_gradient, offset - learning_rate * offset_gradient


# Static learning rate
lr = 0.8e-1

# Starting slope and offset
a, b = -4, 0

# The point that should be predicted
X, Y = 3, 1

slope_range = np.arange(-5, 5, 0.1)
x_range = np.arange(-5, 5, 0.1)

fig = plt.figure(figsize=(12, 6), dpi=80)
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], projection="3d", computed_zorder=False)

ax1.axhline(0, color='black', linewidth=.5)
ax1.axvline(0, color='black', linewidth=.5)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)

# The plot of the line
line = [f(a, b, x) for x in x_range]
ax1.scatter(X, Y)
ax1.plot(x_range, line)
length = sqrt(1.0+a**2)
ax1.arrow(x=0, y=b, dx=1/length, dy=a/length, width=5e-2)

# Loss function as a function of slope
surface_X = np.arange(-5, 5, 0.1)
surface_Y = surface_X
surface_X, surface_Y = np.meshgrid(surface_X, surface_Y)
Z = np.copy(surface_X)
for i in range(surface_X.shape[0]):
    for j in range(surface_X.shape[1]):
        loss = calc_loss(slope=surface_X[i, j], offset=surface_Y[i, j], x=X, y=Y)
        Z[i, j] = loss

surf = ax2.plot_surface(surface_X, surface_Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=1)

# Loss at starting state
loss = calc_loss(a, b, X, Y)
ax2.scatter(xs=a, ys=b, zs=loss, marker="o", s=10)

fname = "./img_2_DOG/0.png"
fig.savefig("./img_2_DOG/0.png")
filenames = [fname]

# Gradient updates
for i in range(1, 20):
    ax1.lines.pop(-1)

    new_a, new_b = gradient_step(a, b, X, Y, lr)
    loss = calc_loss(new_a, new_b, X, Y)
    ax2.scatter(xs=new_a, ys=new_b, zs=loss, marker="o", s=10)

    line_x = np.linspace(a, new_a, num=10)
    line_y = np.linspace(b, new_b, num=10)
    line_z = np.linspace(calc_loss(a, b, X, Y), calc_loss(new_a, new_b, X, Y), num=10)
    ax2.plot(xs=line_x, ys=line_y, zs=line_z, linestyle="dotted")

    line = [f(new_a, new_b, x) for x in x_range]
    ax1.plot(x_range, line)

    length = sqrt(1.0+new_a**2)
    ax1.arrow(x=0, y=new_b, dx=1/length, dy=new_a/length, width=5e-2)

    fname = "./img_2_DOG/%s.png" % i
    fig.savefig(fname)
    filenames.append(fname)

    a, b = new_a, new_b

images = []
for filename in filenames:
    images.append(imageio.v2.imread(filename))
imageio.mimsave('../assets/gradient_2_dog.gif', images, "GIF", duration=1e3, loop=0)