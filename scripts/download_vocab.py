import os
import subprocess

def download_vocab(file_path, filename):
    if not os.path.isfile(file_path):
        print('File does not exist, downloading new one.')

        download_command = f'curl https://s3.amazonaws.com/models.huggingface.co/bert/{filename}.txt --output {file_path}'

        subprocess.run(download_command.split())
        print('New file downloaded.')

    else:
        print('File already exists, no changes made.')
