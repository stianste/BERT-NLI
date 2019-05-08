import matplotlib
import matplotlib.pyplot as plt

# Data for plotting
learning_rates = [2e-5, 3e-5, 4e-5, 5e-5]
epochs = ['3', '4', '5']
accs_2 = [0.749, 0.760, 0.759]
accs_3 = [0.746, 0.765, 0.777]
accs_4 = [0.736, 0.760, 0.761]
accs_5 = [0.734, 0.746, 0.765]

fig, ax = plt.subplots()
ax.plot(epochs, accs_2)
ax.plot(epochs, accs_3)
ax.plot(epochs, accs_4)
ax.plot(epochs, accs_5)

ax.set(xlabel='Number of epochs', ylabel='Accuracy',
       title='Accuracies Obtained Over Epochs For Different Learning Rates.')

plt.legend(learning_rates, loc='upper left', title='Learning Rate')

fig.savefig("plots/bert_base_accuracies.png")
plt.show()