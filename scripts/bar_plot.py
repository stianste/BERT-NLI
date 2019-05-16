import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
model_type = 'svm'
n_groups = 6

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.6

svm_5000 =  [0.549, 0.700, 0.728, 0.715, 0.633, 0.517]
svm_10000 = [0.549, 0.704, 0.744, 0.728, 0.695, 0.570]
svm_30000 = [0.549, 0.703, 0.759, 0.731, 0.731, 0.608]
svm_inf =   [0.549, 0.703, 0.758, 0.727, 0.729, 0.590]

ffnn_5000 =  [0.566, 0.726, 0.735, 0.732, 0.667, 0.507]
ffnn_10000 = [0.561, 0.735, 0.758, 0.755, 0.715, 0.578]
ffnn_30000 = [0.564, 0.743, 0.789, 0.761, 0.766, 0.647]
ffnn_inf =   [0.569, 0.733, 0.795, 0.757, 0.775, 0.655]

if model_type == 'svm':
    bar_5000 = ax.bar(index - bar_width, svm_5000, bar_width,
                alpha=opacity,
                label='5000')
    bar_10000 = ax.bar(index, svm_10000, bar_width,
                alpha=opacity,
                label='10000')
    bar_30000 = ax.bar(index + bar_width, svm_30000, bar_width,
                alpha=opacity,
                label='30000')
    bar_inf = ax.bar(index + 2 * bar_width, svm_inf, bar_width,
                    alpha=opacity,
                    label='inf')

elif model_type == 'ffnn':
    bar_5000 = ax.bar(index - bar_width, ffnn_5000, bar_width,
                alpha=opacity,
                label='5000')
    bar_10000 = ax.bar(index, ffnn_10000, bar_width,
                alpha=opacity,
                label='10000')
    bar_30000 = ax.bar(index + bar_width, ffnn_30000, bar_width,
                alpha=opacity,
                label='30000')
    bar_inf = ax.bar(index + 2 * bar_width, ffnn_inf, bar_width,
                    alpha=opacity,
                    label='inf')


ax.set_ylim(0.5)
ax.set_xlabel('Feature Type')
ax.set_ylabel('Test Accuracy')
ax.set_title(f'{model_type.upper()} Accuracies for Different Feature Types')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('char2', 'char3', 'char4', 'word1', 'word2', 'word3'))
ax.legend(title='Max features')

fig.tight_layout()
plt.show()

fig.savefig(f'plots/{model_type}_accs.png')
plt.show()