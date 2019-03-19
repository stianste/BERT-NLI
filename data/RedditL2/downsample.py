import os
import random
import logging
from collections import Counter, defaultdict

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

random.seed(1)

def downsample_reddit_data(data_dir: str, median_chunks: int, label2language: dict) -> None:
    # Each class must have same number of users.
    # Find number of users tagged with each label in the data,
    # then randomly select the minimum number from each class.
    # Number of chunks is still not equal, so cap it at the median. Ish 3, vs 17

    logger.info(f'Downsampling {data_dir}')

    unique_users_per_language_counter = Counter()
    user2chunks = defaultdict(list)
    user2label = {}

    prefix = 'text_chunks'

    for language_folder in os.listdir(f'{prefix}/{data_dir}'):
        label = language_folder.split('.')[1]
        for username in os.listdir(f'{prefix}/{data_dir}/{language_folder}'):
            for chunk in os.listdir(f'{prefix}/{data_dir}/{language_folder}/{username}'):
                with open(os.path.join(prefix, data_dir, language_folder, username, chunk), 'r') as f:
                    if label == 'Ukraine':
                        continue # Ignore Ukraine from now, as it is not included in the original article
                    text = ''.join(f.readlines()).lower()

                    language = label2language[label]
                    if not username in user2chunks:
                        unique_users_per_language_counter[language] += 1
                    user2chunks[username].append(text)
                    user2label[username] = language

    logger.info(unique_users_per_language_counter)

    max_num_users = min([unique_users_per_language_counter[language] for 
                          language in unique_users_per_language_counter.keys()])

    logger.info(f'The language with the least unique users has {max_num_users} users')

    prefix = 'reddit_downsampled'

    one_user_has_more_than_one_chunk = False

    for username in user2chunks.keys():
        user_label = user2label[username]

        language_folder = f'{prefix}/{data_dir}/{user_label}'
        if not os.path.exists(language_folder):
            os.makedirs(language_folder)

        num_users_for_language = len([name for name in os.listdir(language_folder) 
                                            if os.path.isfile(name)])

        if num_users_for_language >= max_num_users:
            logger.info(f'{user_label} now has {max_num_users} users. Not addind more.')

        user_chunks = user2chunks[username]
        user_chunks = random.sample(user_chunks, min(median_chunks, len(user_chunks)))

        num_chunks = len(user_chunks)
        if num_chunks > 1:
            logger.info(f'{username} has {num_chunks} chunks.')
            one_user_has_more_than_one_chunk = True


        user_folder = f'{prefix}/{data_dir}/{user_label}/{username}'
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        for chunk_num, chunk in enumerate(user_chunks):
            with open(f'{user_folder}/chunk{chunk_num}', 'w') as f:
                f.write(chunk)

    if not one_user_has_more_than_one_chunk:
        logger.warning('All users only had 1 chunk.')

def main():
    logger.info('Running!')
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
