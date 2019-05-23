import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))

x_axis = np.arange(-3, 3, 0.1)
tan_h = [np.tanh(x) for x in x_axis]
sig = [sigmoid(x) for x in x_axis]
rel = [relu(x) for x in x_axis]
gel = [gelu(x) for x in x_axis]

names = ['tanh', 'Sigmoid', 'ReLU', 'GELU']

fig, ax = plt.subplots()
ax.plot(x_axis, tan_h, alpha=0.8)
ax.plot(x_axis, sig, alpha=0.8)
ax.plot(x_axis, rel, alpha=0.8)
ax.plot(x_axis, gel, alpha=0.8)

ax.set(title='Activation Functions')

plt.legend(names, loc='upper left', title='Activation Function')

fig.savefig("plots/activation_functions.png")
plt.show()