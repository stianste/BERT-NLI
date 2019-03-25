import os

import sys
sys.path.append('./..')

from argparse import ArgumentParser
from collections import defaultdict

from run_BERT_NLI import get_eval_folder_name

parser = ArgumentParser()


parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--output_dir",
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--cross_k",
                    type=int,
                    default=10
                    help="The number of folds to use for cross validation.")
parser.add_argument("--full_path",
                    type=str,
                    default=None,
                    help="If running manually, this will be used as the full path to the folder.")

args = parser.parse_args()

eval_foldername = get_eval_folder_name(args)

if args.full_path:
    full_path = args.full_path
else:
    full_path = os.path.join(args.output_dir, eval_foldername)

results = defaultdict(float)

for fold_result_file in os.listdir(full_path):
    with open(os.path.join(full_path, fold_result_file), 'r') as f:
        for line in f.readlines():
            key, _, value = line.split()
            results[key] += float(value)

# Average results over k-folds and save to file
for key in results.keys():
    results[key] /= args.cross_k

# Get overall f1 score.
precision, recall = results['precision'], results['recall']
results['f1'] = 2 * (precision * recall) / (precision + recall)

with open(os.path.join(full_path, 'final.txt'), 'w') as f:
    for key in sorted(results.keys()):
        value = results[key]
        f.write(f'{key} = {value}\n')

