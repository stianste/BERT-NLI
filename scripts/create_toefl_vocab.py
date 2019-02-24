from nltk.tokenize import word_tokenize

vocab = set()

base_path = './data/NLI-shared-task-2017'

with open(f'{base_path}/all.txt', 'r') as f:
    for line in f.readlines():
        words = word_tokenize(line)
        for word in words:
            vocab.add(word)

with open(f'{base_path}/vocab.txt', 'w') as f:
    for word in vocab:
        f.write(word)
        f.write('\n')
