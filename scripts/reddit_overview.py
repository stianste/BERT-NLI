import os
from collections import defaultdict

label2language = {
    "Austria" : "German",
    "Germany" : "German",
    "Australia" : "English",
    "Ireland" : "English",
    "NewZealand" : "English",
    "UK" : "English",
    "US" : "English",
    "Bulgaria" : "Bulgarian",
    "Croatia" : "Croatian",
    "Czech" : "Czech",
    "Estonia" : "Estonian",
    "Finland" : "Finish",
    "France" : "French",
    "Greece" : "Greek",
    "Hungary" : "Hungarian",
    "Italy" : "Italian",
    "Lithuania" : "Lithuanian",
    "Netherlands" : "Dutch",
    "Norway" : "Norwegian",
    "Poland" : "Polish",
    "Portugal" : "Portuguese",
    "Romania" : "Romanian",
    "Russia" : "Russian",
    "Serbia" : "Serbian",
    "Slovenia" : "Slovenian",
    "Spain" : "Spanish",
    "Mexico" : "Spanish",
    "Sweden" : "Swedish",
    "Turkey" : "Turkish",
}

def split_text_chunk_lines(text_lines, max_seq_len=512):
    '''
    Naivley splits a reddit text chunk into multiple text chunks of size ~512 tokens
    '''
    chunks = []
    sentence_words = []
    for sentence in text_lines:
        words = sentence.split()
        if len(sentence_words) + len(words) > max_seq_len:
            chunks.append(' '.join([word.lower() for word in sentence_words]))
            sentence_words = []

        sentence_words += words

    chunks.append(' '.join([word.lower() for word in sentence_words]))
    return chunks

def discover_examples(data_dir: str):
    users = set()
    num_chunks = 0
    num_sub_chunks = 0
    lang2num_chunks = defaultdict(int)
    lang2num_sub_chunks = defaultdict(int)

    for language_folder in os.listdir(data_dir):
        if language_folder.split('.')[1] == 'Ukraine':
            continue

        language = label2language[language_folder.split('.')[1]]

        for username in os.listdir(f'{data_dir}/{language_folder}'):
            users.add(username)
            for chunk in os.listdir(f'{data_dir}/{language_folder}/{username}'):
                num_chunks += 1
                lang2num_chunks[language] += 1

                full_path = f'{data_dir}/{language_folder}/{username}/{chunk}' 
                with open(full_path, 'r') as f:
                    sub_chunks = split_text_chunk_lines(f.readlines())
                    num_sub_chunks += len(sub_chunks)
                    lang2num_sub_chunks[language] += len(sub_chunks)

    return users, num_chunks, num_sub_chunks, lang2num_chunks, lang2num_sub_chunks

users, num_chunks, num_sub_chunks, lang2num_chunks, lang2num_sub_chunks = discover_examples('./data/RedditL2/text_chunks/europe_data')

print('Number of europe users:', len(users))
print('Number of chunks:', num_chunks)
print('Number of sub-chunks:', num_sub_chunks)
print()

print('-'*50)
for lang in lang2num_chunks.keys():
    print(f'{lang:<10} | {lang2num_chunks[lang]:>10} | {lang2num_sub_chunks[lang]:>10} |')
print('-'*50, '\n')

users, num_chunks, num_sub_chunks, lang2num_chunks, lang2num_sub_chunks = discover_examples('./data/RedditL2/text_chunks/non_europe_data')

print('Number of non europe users:', len(users))
print('Number of chunks:', num_chunks)
print('Number of sub-chunks:', num_sub_chunks)

print('-'*50)
for lang in lang2num_chunks.keys():
    print(f'{lang:<10} | {lang2num_chunks[lang]:>10} | {lang2num_sub_chunks[lang]:>10} |')
print('-'*50, '\n')