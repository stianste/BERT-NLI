import re
from os import path
from download_vocab import download_vocab
from spellchecker import SpellChecker
from collections import Counter
from nltk.tokenize import word_tokenize

# 2 - 100 [mask ...] 105 - 999

bert_vocab = []

bert_vocab_file_name = 'bert-large-uncased-vocab'
bert_vocab_path = f'./data/vocabs/{bert_vocab_file_name}.txt'

base_path = 'data/NLI-shared-task-2017'
model_prefix = base_path + '/toefl'

idx_to_fill = [x for x in range(1, 100)] + [y for y in range(104, 999)]

if not path.isfile(bert_vocab_path):
    download_vocab(bert_vocab_path, bert_vocab_file_name)

with open(bert_vocab_path, 'r') as bert_vocab_file:
    for word in bert_vocab_file.readlines():
        bert_vocab.append(word)

print('BERT vocab size:', len(bert_vocab))

misspelled_counter = Counter()
misspelled_words = set()
spell_checker = SpellChecker()

with open(f'{base_path}/all_lower.txt', 'r') as model_text_file:
    for sentence in model_text_file.readlines():
        words = word_tokenize(sentence)
        words = [re.sub('\W+', '', word) for word in words]
        misspelled = spell_checker.unknown(words)
        for word in misspelled:
            if word: # Ignore empty string
                misspelled_words.add(word)
                misspelled_counter[word] += 1

misspelled_word_counts = misspelled_counter.most_common()
print("Most common mispelled words:", misspelled_word_counts[:10])
misspelled_word_counts.reverse()

words_intersecting = set(bert_vocab).intersection(misspelled_words)
print("Alot:", "alot" in words_intersecting)

for idx in idx_to_fill:
    word = misspelled_word_counts.pop()[0]
    if idx < 10:
        print("Considering", word)
    while word in words_intersecting:
        word = misspelled_word_counts.pop()

    if idx < 10:
        print("Adding", word)

    bert_vocab[idx] = word + "\n"

with open(bert_vocab_path.split('.txt')[0] + '_filled_misspelled.txt', 'w') as f:
    for word in bert_vocab:
        f.write(word)
