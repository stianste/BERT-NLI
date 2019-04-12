import os
from pytorch_pretrained_bert.tokenization import BertTokenizer

def get_toefl_token_sizes():
    token_sizes = []
    full_path = './data/NLI-shared-task-2017/data/essays/train/original/'

    for filename in os.listdir(full_path):
        with open(os.path.join(full_path, filename), 'r') as f:
            text = ''.join(f.readlines()).lower()

            tokens = tokenizer.tokenize(text)
            token_sizes.append(len(tokens))

    return token_sizes

def get_reddit_token_sizes():
    token_sizes = []
    data_dir = './data/RedditL2/text_chunks/europe_data'

    for language_folder in os.listdir(data_dir):
        for username in os.listdir(f'{data_dir}/{language_folder}'):
            for chunk in os.listdir(f'{data_dir}/{language_folder}/{username}'):
                full_path = f'{data_dir}/{language_folder}/{username}/{chunk}'
                with open(full_path, 'r') as f:
                    text = ''.join(f.readlines()).lower()

                    tokens = tokenizer.tokenize(text)
                    token_sizes.append(len(tokens))

    return token_sizes

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
token_sizes = get_reddit_token_sizes()

print('Max token size', max(token_sizes)) # 910, 747

avg_size = sum(token_sizes) // len(token_sizes)
print('Avg token size', avg_size) # 369, 365

num_above_average =  len([x for x in token_sizes if x > avg_size])
print('Num token size > avg', num_above_average)
print('Percentage', num_above_average / len(token_sizes))

n = 512
num_above_n =  len([x for x in token_sizes if x > n])
print(f'Num token size > {n}', num_above_n)
print('Percentage', num_above_n / len(token_sizes))