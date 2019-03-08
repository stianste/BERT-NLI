from os import path
from download_vocab import download_vocab

vocab = set()
custom_vocab_size = 16000
bert_vocab_file_name = 'bert-large-uncased-vocab'
bert_vocab_path = f'./data/vocabs/{bert_vocab_file_name}.txt'

if not path.isfile(bert_vocab_path):
    download_vocab(bert_vocab_path, bert_vocab_file_name)

with open (bert_vocab_path, 'r') as bert_vocab_file:
    for word in bert_vocab_file.readlines():
        vocab.add(word)

with open (f'./data/NLI-shared-task-2017/toefl_wordpiece_size_{custom_vocab_size}.vocab', 'r') as custom_vocab_file:
    for word in custom_vocab_file.readlines():
        vocab.add(word)

print('Total vocab size is:', len(vocab))

new_file_path = f'./data/vocabs/{bert_vocab_file_name}_fuzed_{custom_vocab_size}.vocab'

with open (new_file_path, 'w') as new_file:
    for word in vocab:
        new_file.write(f'{word}')
