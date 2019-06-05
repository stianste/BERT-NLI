import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import f1_score
from plot_confusion_matrix import plot_confusion_matrix
from data_processors import RedditInDomainDataProcessor

out_of_domain = True
sub_folder = '/out-of-domain' if out_of_domain else ''

with_bert = True
bert_string = '_wBERT' if with_bert else ''
outputs_folder = './common_predictions/reddit_predictions/out-of-domain/results/outputs/MLPClassifier__ffnn_None_wBERT_'

filenames = sorted(os.listdir(outputs_folder))

accuracies = []
total_num_chunks = 0
all_outputs = []
all_labels = []

for filename in filenames:
    if filename.split('.')[-1] == 'csv':
        print(filename)
        df = pd.read_csv(f'{outputs_folder}/{filename}')
        df['guid'] = df['guid'].apply(lambda guid: '_'.join(guid.split('_')[:-1]))
        chunk2outputs = defaultdict(list)
        chunk2label = {}
        for i, row in df.iterrows():
            guid, label, output = row[0], row[1], row[2]
            chunk2outputs[guid].append(output)
            chunk2label[guid] = label

        majority_votes = {}
        for chunk, outputs in chunk2outputs.items():
            counter = Counter(outputs)
            majority_vote = counter.most_common(1)[0][0]
            majority_votes[chunk] = majority_vote

        num_correct = 0
        for chunk, label in chunk2label.items():
            # print(chunk, majority_votes[chunk], label)
            if majority_votes[chunk] == label:
                num_correct += 1

            all_outputs.append(majority_votes[chunk])
            all_labels.append(label)

        num_chunks = len(chunk2label.keys())
        total_num_chunks += num_chunks
        eval_acc = num_correct / num_chunks
        print(f'{filename:<12}: {num_correct} correct out of {num_chunks}. Accuracy: {eval_acc:.3f}')
        accuracies.append(eval_acc)

other_average = sum([1 for i in range(len(all_outputs)) if all_outputs[i] == all_labels[i]]) / len(all_outputs)
f1 = f1_score(all_outputs, all_labels, average='macro')

average_eval_acc = sum(accuracies) / len(accuracies)
print(f'Average accuracy: {average_eval_acc:.3f}')
print(f'Other average and f1: {other_average:.3f}, {f1:.3f}')
print('Average number of examples:', total_num_chunks / 10)
print(all_labels[:10])
print(all_outputs[:10])

ax = plot_confusion_matrix(all_labels, all_outputs, classes=RedditInDomainDataProcessor(0).get_labels())
ax.set_ylabel('Correct')
ax.set_xlabel('Predicted')
plt.show()
np.set_printoptions(precision=2)
plt.savefig(f'./plots/confusion_matrix.png')
    