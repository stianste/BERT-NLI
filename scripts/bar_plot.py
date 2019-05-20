import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
model_type = 'ffnn'
n_groups = 10
font_size = 14
legend_props = {
    'size' : 20,
}

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.6

svm_5000 =  [0.549, 0.700, 0.728, 0.715, 0.633, 0.517, 0.712, 0.667, 0.405, 0.475]
svm_10000 = [0.549, 0.704, 0.744, 0.728, 0.695, 0.570, 0.707, 0.699, 0.405, 0.482]
svm_30000 = [0.549, 0.703, 0.759, 0.731, 0.731, 0.608, 0.705, 0.740, 0.405, 0.485]
svm_inf =   [0.549, 0.703, 0.758, 0.727, 0.729, 0.590, 0.705, 0.744, 0.405, 0.485]

ffnn_5000 =  [0.566, 0.726, 0.735, 0.732, 0.677, 0.507, 0.720, 0.665, 0.418, 0.446]
ffnn_10000 = [0.561, 0.735, 0.758, 0.755, 0.715, 0.578, 0.743, 0.745, 0.415, 0.465]
ffnn_30000 = [0.564, 0.743, 0.789, 0.761, 0.766, 0.647, 0.745, 0.763, 0.401, 0.476]
ffnn_inf =   [0.569, 0.733, 0.795, 0.757, 0.775, 0.655, 0.742, 0.763, 0.405, 0.482]

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


ax.set_ylim(0.4)
ax.set_xlabel('Feature Type', fontsize=font_size)
ax.set_ylabel('Test Accuracy', fontsize=font_size)
ax.set_title(f'{model_type.upper()} Accuracies for Different Feature Types', fontsize=font_size)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('CHAR2', 'CHAR3', 'CHAR4', 'WORD1', 'WORD2', 'WORD3', 'LEMMA1', 'LEMMA2', 'FUNC1', 'FUNC2'), fontsize=font_size)
legend = ax.legend(title='Max features', prop=legend_props, fontsize=font_size)
legend.get_title().set_fontsize(font_size)

# fig.tight_layout()
plt.show()

fig.savefig(f'plots/{model_type}_accs.png')
plt.show()