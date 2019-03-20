import os
import random
import logging
from collections import Counter, defaultdict
from get_incommon_users import get_incommon_users

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

random.seed(1)

def get_data_from_dir(data_dir: str,
                      label2language: dict,
                      max_chunks_per_user: int,
                      user_list: set, 
                      black_list: set):

    # Each class must have same number of users.
    # Find number of users tagged with each label in the data,
    # then randomly select the minimum number from each class.
    # Number of chunks is still not equal, so cap it at the median. Ish 3, vs 17

    lang2username2chunks = defaultdict(list)

    for label_folder in os.listdir(f'{data_dir}'):
        logger.info(f'Dowsampling {label_folder} in {data_dir}')

        label = label_folder.split('.')[1]
        if label == 'Ukraine':
            continue # Ignore Ukraine from now, as it is not included in the original article
        language = label2language[label]

        for username in os.listdir(f'{data_dir}/{label_folder}'):
            if black_list and username in black_list:
                continue

            if user_list and username not in user_list:
                continue

            user_chunks = []

            for chunk in os.listdir(f'{data_dir}/{label_folder}/{username}'):
                with open(os.path.join(data_dir, label_folder, username, chunk), 'r') as f:
                    text = ''.join(f.readlines()).lower()
                    user_chunks.append(text)

            lang2username2chunks[language].append((username, random.sample(user_chunks, min(max_chunks_per_user, len(user_chunks)))))

    for language, user2chunks_tuple_list in lang2username2chunks.items():
        lang2username2chunks[language] = random.sample(user2chunks_tuple_list, 104)

    sampled_users = set()

    for language, user2chunks_tuple_list in lang2username2chunks.items():
        if not len(user2chunks_tuple_list) == 104:
            logger.warning(
                f'{language} does not have 104 users, it has {len(user2chunks_tuple)}')

        assert len(user2chunks_tuple_list) == 104

        for user2chunks_tuple in user2chunks_tuple_list:
            username = user2chunks_tuple[0]
            sampled_users.add(username)

    logger.info(f'Total number of users are: {len(sampled_users)}, should be {104 * 23}')
    assert len(sampled_users) == 104 * 23 # 104 users for each of the 23 labels

    return lang2username2chunks, sampled_users

def write_user_chunks(data_dir: str, lang2username2chunks) -> None:
    for language, user2chunks_tuple_list in lang2username2chunks.items():
        language_folder = f'{data_dir}/{language}'

        if not os.path.exists(language_folder):
            os.makedirs(language_folder)

        for user2chunks_tuple in user2chunks_tuple_list:
            username, user_chunks = user2chunks_tuple
            user_folder = f'{language_folder}/{username}'
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)

            for chunk_num, chunk in enumerate(user_chunks):
                with open(f'{user_folder}/chunk{chunk_num + 1}', 'w') as f:
                    f.write(chunk)

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

    europe_folder = './data/RedditL2/reddit_downsampled/europe_data'
    non_europe_folder = './data/RedditL2/reddit_downsampled/non_europe_data'
    os.makedirs(europe_folder, exist_ok=True)
    os.makedirs(non_europe_folder, exist_ok=True)

    black_list = get_incommon_users()

    europe_lang2user2chunks, sampled_users = get_data_from_dir('./data/RedditL2/text_chunks/europe_data',
                                                           label2language, 3,
                                                           black_list=black_list)

    write_user_chunks(europe_folder, europe_lang2user2chunks)

    non_europe_lang2user2chunks, _ = get_data_from_dir('./data/RedditL2/text_chunks/non_europe_data',
                                                   label2language, 17, 
                                                   black_list=black_list,
                                                   user_list=sampled_users)

    write_user_chunks(europe_folder, non_europe_lang2user2chunks)

if __name__ == "__main__":
    main()
