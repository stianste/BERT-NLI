import constants
import os
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

""" 
Creates one big txt file of the entire TOEFL11 training corpus. It does not
use the dev part of the corpus.

Adittionally it will also make one txt consisting of each language, so that
one can easily train BERT on a per-language basis at a later time.
"""

def write_sentences_to_file(sentences: list, f, do_lower=False):
    for sentence in sentences:
        if sentence.strip():
            if do_lower:
                f.write(sentence.strip().lower() + '\n')
            else:
                f.write(sentence.strip() + '\n')

    f.write('\n') # Separate documents by new line

dir_path = './data/NLI-shared-task-2017/'
full_path = f'{dir_path}{constants.TOEFL11_TRAINING_DATA_PATH}'

filenames = [filename for filename in os.listdir(full_path)]
do_lower = True

id2label = {}

with open(f'{dir_path}{constants.TOEFL11_TRAINING_LABELS_LOCATION}', 'r') as f:
    for line in f.readlines():
        example_id, _, _, label = line.split(',')
        id2label[example_id.strip()] = label.strip()

if do_lower:
    file_to_write = f'{dir_path}all_lower.txt'
else:
    file_to_write = f'{dir_path}all.txt'

with open (file_to_write, "w") as main_file:
    for filename in filenames:
        with open(os.path.join(full_path, filename), "r") as f:
            sentences = sent_tokenize(" ".join([line.strip() for line in f.readlines()]))
            write_sentences_to_file(sentences, main_file, do_lower=do_lower)

            example_id = filename.split('.')[0]
            label = id2label[example_id]
            with open(f'{dir_path}{label}.txt', 'a+') as lang_file:
                write_sentences_to_file(sentences, lang_file, do_lower=do_lower)
