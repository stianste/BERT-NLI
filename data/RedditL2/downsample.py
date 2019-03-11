import os
import random
from collections import Counter, defaultdict

random.seed(1)

def downsample_reddit_data(data_dir: str, median_chunks: int, label2language: dict) -> None:
    # Each class must have same number of users.
    # Find number of users tagged with each label in the data,
    # then randomly select the minimum number from each class.
    # Number of chunks is still not equal, so cap it at the median. Ish 3, vs 17

    unique_users_per_language_counter = Counter()
    user2chunks = defaultdict(list)
    user2label = {}

    prefix = 'text_chunks'

    for language_folder in os.listdir(f'{prefix}/{data_dir}'):
        label = language_folder.split('.')[1]
        for username in os.listdir(f'{prefix}/{data_dir}/{language_folder}'):
            for chunk in os.listdir(f'{prefix}/{data_dir}/{language_folder}/{username}'):
                with open(os.path.join(prefix, data_dir, language_folder, username, chunk), 'r') as f:
                    text = ''.join(f.readlines()).lower()
                    language = label2language[label]
                    if not username in user2chunks:
                        unique_users_per_language_counter[language] += 1
                    user2chunks[username].append(text)
                    user2label[username] = language

    print(unique_users_per_language_counter)

    max_num_users = min([unique_users_per_language_counter[language] for 
                          language in unique_users_per_language_counter.keys()])

    print('The maximum number of users for a language is:', max_num_users)

    prefix = 'reddit_downsampled'

    for username in user2chunks.keys():
        user_label = user2label[username]
        user_chunks = user2chunks[username]
        user_chunks = random.sample(user_chunks, min(median_chunks, len(user_chunks)))

        language_folder = f'{prefix}/{data_dir}/{user_label}'
        if not os.path.exists(language_folder):
            os.makedirs(language_folder)

        user_folder = f'{prefix}/{data_dir}/{user_label}/{username}'
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        for chunk_num, chunk in enumerate(user_chunks):
            with open(f'{user_folder}/chunk{chunk_num}', 'w') as f:
                f.write(chunk)

def main():
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

    europe_path = 'europe_data'
    non_europe_path = 'non_europe_data'

    os.makedirs(f'reddit_downsampled/{europe_path}', exist_ok=True)
    os.makedirs(f'reddit_downsampled/{non_europe_path}', exist_ok=True)

    downsample_reddit_data(europe_path, 3, label2language)
    downsample_reddit_data(non_europe_path, 17, label2language)

if __name__ == "__main__":
    main()