import os
import logging
from collections import defaultdict

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def get_users_from_dir(data_dir: str):
    lang2usernames = defaultdict(set)
    usernames = set()

    for label_folder in os.listdir(f'{data_dir}'):
        label = label_folder.split('.')[1]
        if label == 'Ukraine':
            continue # Ignore Ukraine from now, as it is not included in the original article

        for username in os.listdir(f'{data_dir}/{label_folder}'):
            usernames.add(username)
            lang2usernames[label].add(username)

    return usernames, lang2usernames

def get_incommon_users(compare_languages=False):
    europe_usernames, europe_lang2usernames = get_users_from_dir(
        './data/RedditL2/text_chunks/europe_data')
    non_europe_usernames, non_europe_lang2usernames = get_users_from_dir(
        './data/RedditL2/text_chunks/non_europe_data')

    incommon_users = set()

    for username in europe_usernames:
        if username not in non_europe_usernames:
            incommon_users.add(username)

    logger.info(f'Total number of Europe users: {len(europe_usernames)}')
    logger.info(f'Total number of non Europe users: {len(non_europe_usernames)}')
    logger.info(f'Total number of missing users: {len(incommon_users)}')

    if compare_languages:
        for language, userset in europe_lang2usernames.items():
            non_europe_user_set = non_europe_lang2usernames[language]
            common = userset.intersection(non_europe_user_set)
            logger.info(f'{language} has {len(common)} in and out of domain.')

    return incommon_users

if __name__ == "__main__":
    get_incommon_users()
