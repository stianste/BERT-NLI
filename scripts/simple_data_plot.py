import matplotlib
import matplotlib.pyplot as plt

# Data for plotting
epochs = [3, 4, 5, 10, 20]
eval_loss = [
       1.189,
       1.318,
       1.448,
       1.848,
       2.208
]
train_loss = [
       0.496,
       0.205,
       0.082,
       0.003,
       0.000
]

fig, ax = plt.subplots()
ax.plot(epochs, train_loss)
ax.plot(epochs, eval_loss)

ax.set(xlabel='Number of epochs', ylabel='Loss',
       title='Training and evaluation loss per number of epochs')

plt.legend(['Training loss', 'Evaluation Loss'], loc='upper left')

fig.savefig("plots/bert_base_loss.png")
plt.show()