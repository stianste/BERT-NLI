import os

def verify_dir(data_dir, is_europe=False):
    total_num_users = 0
    total_num_chunks = 0
    for language_folder in os.listdir(data_dir):
        language = language_folder
        dir_iterator = os.listdir(f'{data_dir}/{language_folder}')
        num_users = len(dir_iterator)
        total_num_users += num_users
        if is_europe:
            assert num_users == 104, f'{language} has {len(dir_iterator)} users in europe.' 

        for username in dir_iterator:
            num_chunks = len(os.listdir(f'{data_dir}/{language_folder}/{username}'))
            total_num_chunks += num_chunks

    assert total_num_users < total_num_chunks, 'All user folders only contains one chunk.'

    return total_num_users, total_num_chunks


def main():
    num_europe_users, europe_total_num_chunks = verify_dir(
        './data/RedditL2/reddit_downsampled/europe_data', is_europe=True)
    num_non_europe_users, non_europe_total_num_chunks = verify_dir(
        './data/RedditL2/reddit_downsampled/non_europe_data')

    print('Num europe users:', num_europe_users, '. Num chunks:', europe_total_num_chunks)
    print('Num non europe users:', num_non_europe_users, '. Num chunks:', non_europe_total_num_chunks)

if __name__ == "__main__":
    main()
