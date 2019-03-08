import sentencepiece as spm
from os import path
from download_vocab import download_vocab

# 2 - 100 [mask ...] 105 - 999

bert_vocab = []

bert_vocab_file_name = 'bert-large-uncased-vocab'
bert_vocab_path = f'./data/vocabs/{bert_vocab_file_name}.txt'

base_path = 'data/NLI-shared-task-2017'
model_prefix = base_path + '/toefl'

idx_to_fill = set([x for x in range(1, 100)] + [y for y in range(104, 999)])
vocab_size = 4125

if not path.isfile(bert_vocab_path):
    download_vocab(bert_vocab_path, bert_vocab_file_name)

with open(bert_vocab_path, 'r') as bert_vocab_file:
    for word in bert_vocab_file.readlines():
        bert_vocab.append(word)

print('BERT vocab size:', len(bert_vocab))

custom_vocab = set()
custom_vocab_filename = f'toefl_wordpiece_size_{vocab_size}.vocab'
with open(f'{base_path}/{custom_vocab_filename}', 'r') as model_vocab_file:
    for word in model_vocab_file.readlines():
        custom_vocab.add(word)

print('Custom vocab size:', len(custom_vocab))

words_intersecting = set(bert_vocab).intersection(custom_vocab)
num_intersecting = len(words_intersecting)
words_not_intersecting = custom_vocab - words_intersecting

print('Num intersect:', num_intersecting)
print('Num not intersecting:', len(words_not_intersecting))
print('Num needing replacement:', len(idx_to_fill))
print('Num left:', len(words_not_intersecting) - len(idx_to_fill))

for idx in idx_to_fill:
    bert_vocab[idx] = words_not_intersecting.pop()

assert len(words_not_intersecting) == 0

with open(bert_vocab_path.split('.txt')[0] + '_filled.txt', 'w') as f:
    for word in bert_vocab:
        f.write(word)