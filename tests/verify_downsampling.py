import os

def verify_dir(data_dir):
    for language_folder in os.listdir(data_dir):
        language = language_folder
        dir_iterator = os.listdir(f'{data_dir}/{language_folder}')
        assert len(dir_iterator) == 104, f'{language} has {len(dir_iterator)} users.' 

        for username in dir_iterator:
            num_chunks = len(os.listdir(f'{data_dir}/{language_folder}/{username}'))
            if len(num_chunks) > 1:
                return

    assert False, 'All user folders only contains one chunk.'


def main():
    verify_dir('./data/RedditL2/reddit_downsampled/europe_data')
    verify_dir('./data/RedditL2/reddit_downsampled/non_europe_data')

if __name__ == "__main__":
    main()
