import os
from pytorch_pretrained_bert.tokenization import BertTokenizer

full_path = './data/NLI-shared-task-2017/data/essays/train/original/'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

token_sizes = []

for filename in os.listdir(full_path):
    example_id = filename.split('.')[0]
    with open(os.path.join(full_path, filename), "r") as f:
        text = "".join(f.readlines()).lower()

        tokens = tokenizer.tokenize(text)
        token_sizes.append(len(tokens))

print("Max token size", max(token_sizes))
print("Avg token size", sum(token_sizes) // len(token_sizes))