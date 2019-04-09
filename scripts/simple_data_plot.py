import matplotlib
import matplotlib.pyplot as plt

# Data for plotting
epochs = [3, 4, 5, 10, 20]
eval_loss = [
       1.258,
       1.423,
       1.673,
       2.177,
       2.601
]
train_loss = [
       0.405,
       0.192,
       0.067,
       0.005,
       0.002
]

fig, ax = plt.subplots()
ax.plot(epochs, train_loss)
ax.plot(epochs, eval_loss)

ax.set(xlabel='Number of epochs', ylabel='Loss',
       title='Training and evaluation loss per number of epochs')

plt.legend(['Training loss', 'Evaluation Loss'], loc='upper left')

fig.savefig("plots/bert_base_loss.png")
plt.show()