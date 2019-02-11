import constants
import os

""" 
Creates one big txt file of the entire TOEFL11 training corpus. It does not
use the dev part of the corpus.

Adittionally it will also make one txt consisting of each language, so that
one can easily train BERT on a per-language basis at a later time.

"""

dir_path = './data/NLI-shared-task-2017/'
full_path = f'{dir_path}{constants.TOEFL11_TRAINING_DATA_PATH}'

filenames = [filename for filename in os.listdir(full_path)]

with open (f'{dir_path}all.txt', "w") as main_file:
    for filename in filenames:
        with open(os.path.join(full_path, filename), "r") as f:
            sentences = "".join(f.readlines()).split('.')
            for sentence in sentences:
                if sentence.strip():
                    main_file.write(sentence.strip() + '.\n')

            main_file.write('\n') # Separate docs by newline
