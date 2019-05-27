import os
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import f1_score

out_of_domain = False
sub_folder = '/out-of-domain' if out_of_domain else ''
outputs_folder = f'./results/reddit{sub_folder}/seq_512_batch_16_epochs_5.0_lr_3e-05'

with_bert = True
bert_string = '_wBERT' if with_bert else ''
# outputs_folder = f'./common_predictions/reddit_predictions/results/outputs/MLPClassifier__ffnn_30000{bert_string}_/'
# outputs_folder = './results/reddit/out_to_in_domain/' # seq_512_batch_16_epochs_5.0_lr_3e-05'
# filenames = ['05-25_10:07_seq_512_batch_16_epochs_5.0_lr_3e-05_train_fold_1.csv']
reddit_out_to_in_domain = './results/reddit/out_to_in_domain/seq_512_batch_16_epochs_5.0_lr_3e-05/fold_1.csv'
# oracle_df = pd.read_csv()
# oracle_df['guid'] = oracle_df['guid'].apply(lambda guid: '_'.join(guid.split('_')[:-1]))

# oracle_predictions = defaultdict(list)
# for i, row in oracle_df.iterrows():
#     oracle_predictions[row.guid].append(row.output)

# oracle_predictions = {guid : Counter(outputs).most_common(1)[0][0] for guid, outputs in oracle_predictions.items()}
# del oracle_df
# filenames = sorted(os.listdir(outputs_folder))
filenames = [reddit_out_to_in_domain]

accuracies = []
total_num_chunks = 0
all_outputs = []
all_labels = []

for filename in filenames:
    if filename.split('.')[-1] == 'csv':
        print(filename)
        df = pd.read_csv(filename)
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
