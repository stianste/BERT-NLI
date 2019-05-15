import os
import pandas as pd
from collections import defaultdict, Counter

out_of_domain = False
sub_folder = 'out-of-domain' if out_of_domain else ''
outputs_folder = f'./results/reddit/{sub_folder}/seq_512_batch_16_epochs_5.0_lr_3e-05/'

accuracies = []
total_num_chunks = 0
for filename in sorted(os.listdir(outputs_folder)):
    if filename.split('.')[-1] == 'csv':
        df = pd.read_csv(f'{outputs_folder}/{filename}')
        df['guid'] = df['guid'].apply(lambda guid: ''.join(guid.split('_')[:-1]))
        chunk2outputs = defaultdict(list)
        chunk2label = {}
        for i, row in df.iterrows():
            guid, label, output, = row[0], row[1], row[2]
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

        num_chunks = len(chunk2label.keys())
        total_num_chunks += num_chunks
        eval_acc = num_correct / num_chunks
        print(f'{filename:<12}: {num_correct} correct out of {num_chunks}. Accuracy: {eval_acc:.3f}')
        accuracies.append(eval_acc)

average_eval_acc = sum(accuracies) / len(accuracies)
print('Average accuracy:', average_eval_acc)
print('Average number of examples:', total_num_chunks / 10)
