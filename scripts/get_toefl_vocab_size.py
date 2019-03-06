from nltk.tokenize import word_tokenize

vocab = set()

base_path = './data/NLI-shared-task-2017'

with open(f'{base_path}/all.txt', 'r') as f:
    for line in f.readlines():
        words = word_tokenize(line)
        for word in words:
            vocab.add(word)

print("Vocab size:", len(vocab)) # 76024