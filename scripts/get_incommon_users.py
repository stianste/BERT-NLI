import os
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def get_users_from_dir(data_dir: str):
    usernames = set()

    for label_folder in os.listdir(f'{data_dir}'):
        label = label_folder.split('.')[1]
        if label == 'Ukraine':
            continue # Ignore Ukraine from now, as it is not included in the original article

        for username in os.listdir(f'{data_dir}/{label_folder}'):
            usernames.add(username)

    return usernames

def get_incommon_users():
    europe_usernames = get_users_from_dir('./data/RedditL2/text_chunks/europe_data')
    non_europe_usernames = get_users_from_dir('./data/RedditL2/text_chunks/non_europe_data')

    incommon_users = set()

    for username in europe_usernames:
        if username not in non_europe_usernames:
            incommon_users.add(username)

    logger.info(f'Total number of Europe users: {len(europe_usernames)}')
    logger.info(f'Total number of non Europe users: {len(non_europe_usernames)}')
    logger.info(f'Total number of missing users: {len(incommon_users)}')

    return incommon_users

if __name__ == "__main__":
    get_incommon_users()
