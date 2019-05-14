import constants
import os
import random
from collections import defaultdict
from sklearn.model_selection import KFold

from typing import List

def split_text_chunk_lines(text_lines, max_seq_len=512):
    '''
    Naivley splits a reddit text chunk into multiple text chunks of size ~512 tokens
    '''
    chunks = []
    sentence_words = []
    for sentence in text_lines:
        words = sentence.split()
        if len(sentence_words) + len(words) > max_seq_len:
            chunks.append(' '.join([word.lower() for word in sentence_words]))
            sentence_words = []

        sentence_words += words

    chunks.append(' '.join([word.lower() for word in sentence_words]))
    return chunks


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _get_dev_fold(self, examples, fold_number, fold_size):
        return examples[fold_number * fold_size : (fold_number + 1) * fold_size]

    def _get_train_fold(self, examples, fold_number, fold_size):
        return examples[0:fold_number * fold_size] + examples[(fold_number + 1) * fold_size : ]


class TOEFL11Processor(DataProcessor):
    """Processor for the TOEFL11 data set."""

    def __init__(self):
        self.id2label = {}

    def get_train_examples(self, data_dir: str='./data/NLI-shared-task-2017/') -> List[InputExample]:
        full_path = data_dir + constants.TOEFL11_TRAINING_DATA_PATH
        self._create_labels(data_dir)
        return self._get_examples(full_path)

    def get_dev_examples(self, data_dir: str='./data/NLI-shared-task-2017/'):
        full_path = data_dir + constants.TOEFL11_DEV_DATA_PATH
        return self._get_examples(full_path)

    def get_labels(self):

        labels = list(set([
            "HIN",
            "ARA",
            "JPN",
            "GER",
            "TEL",
            "KOR",
            "SPA",
            "ITA",
            "CHI",
            "FRE",
            "TUR",
        ]))

        assert len(labels) == 11, f'TOEFL11 Should have 11 labels. Actual: {len(labels)}'

        return sorted(labels)

    def _get_examples(self, full_path: str) -> List[InputExample]:
        examples = []

        for filename in os.listdir(full_path):
            example_id = filename.split('.')[0]
            with open(os.path.join(full_path, filename), "r") as f:
                label = self.id2label[example_id]

                text = "".join(f.readlines()).lower()

                example = InputExample(guid=example_id, text_a=text, label=label)
                examples.append(example)

        return examples

    def _create_labels(self, data_dir):
        training_labels_filepath = data_dir + constants.TOEFL11_TRAINING_LABELS_LOCATION
        dev_labels_filepath = data_dir + constants.TOEFL11_DEV_LABELS_LOCATION
        self._fill_labels_from_file(training_labels_filepath)
        self._fill_labels_from_file(dev_labels_filepath)

    def _fill_labels_from_file(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            for line in f.readlines():
                example_id, _, _, label = line.split(',')
                self.id2label[example_id.strip()] = label.strip()

class BinaryTOEFL11Processor(TOEFL11Processor):
    """Binary Processor for the TOEFL11 data set."""

    def __init__(self, lang):
        self.id2label = {}
        self.lang = lang

    def get_labels(self):
        return [0, 1]

    def _get_examples(self, full_path: str) -> List[InputExample]:
        examples = []

        for filename in os.listdir(full_path):
            example_id = filename.split('.')[0]
            with open(os.path.join(full_path, filename), "r") as f:
                label = self.id2label[example_id]
                is_correct = label == self.lang

                text = "".join(f.readlines()).lower()
                example_id = filename.split(".")[0]

                example = InputExample(guid=example_id, text_a=text, label=is_correct)
                examples.append(example)

        assert len([1 for example in examples if example.label]) > 0 # At least one example must be true.
        return examples

class RedditInDomainDataProcessor(DataProcessor):
    """Processor for the RedditL2 data set"""

    def __init__(self, fold_number):
        self.fold_number = fold_number - 1
        self.user2examples = {}
        self.lang2usernames = defaultdict(list)
        self.label2language = {
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


    def discover_examples(self, data_dir: str)-> None:
        for language_folder in os.listdir(data_dir):
            language = language_folder
            for username in os.listdir(f'{data_dir}/{language_folder}'):
                self.lang2usernames[language].append(username)
                user_examples = []
                for chunk in os.listdir(f'{data_dir}/{language_folder}/{username}'):
                    full_path = f'{data_dir}/{language_folder}/{username}/{chunk}' 
                    with open(full_path, 'r') as f:
                        sub_chunks = split_text_chunk_lines(f.readlines())

                        for i, sub_chunk in enumerate(sub_chunks):
                            user_examples.append(
                                InputExample(guid=f'{username}_{chunk}_{i}', text_a=sub_chunk, label=language)
                            )

                self.user2examples[username] = user_examples

        for language, user_list in self.lang2usernames.items():
            assert len(set(user_list)) == 104, f'{language} has {len(user_list)} users.'

    def _get_examples_for_fold(self, fold_function):
        examples = []
        fold_size = int(104 / 10)

        for usernames in self.lang2usernames.values():
            usernames_for_lang = fold_function(usernames, self.fold_number, fold_size)
            for username in usernames_for_lang:
                for example in self.user2examples[username]:
                    examples.append(example)

        return examples

    def get_train_examples(self, data_dir: str='./data/RedditL2/reddit_downsampled/europe_data') -> List[InputExample]:
        '''
        Beware when calling this function in succession. Running the function twice will rediscover
        the examples and yield all examples twice.
        '''
        self.discover_examples(data_dir)
        return self._get_examples_for_fold(self._get_train_fold)

    def get_dev_examples(self, data_dir: str='./data/RedditL2/reddit_downsampled/europe_data'):
        return self._get_examples_for_fold(self._get_dev_fold)

    def get_labels(self):
        labels = list(set([
            "Australia",
            "Austria",
            "Bulgaria",
            "Croatia",
            "Czech",
            "Estonia",
            "Finland",
            "France",
            "Germany",
            "Greece",
            "Hungary",
            "Ireland",
            "Italy",
            "Lithuania",
            "Mexico",
            "Netherlands",
            "NewZealand",
            "Norway",
            "Poland",
            "Portugal",
            "Romania",
            "Russia",
            "Serbia",
            "Slovenia",
            "Spain",
            "Sweden",
            "Turkey",
            "UK",
            "US",
        ]))

        for label in labels:
            assert label in self.label2language

        language_list = list(set(self.label2language.values()))
        assert len(language_list) == 23
        return sorted(language_list)

class RedditOutOfDomainDataProcessor(RedditInDomainDataProcessor):
    """Processor for the RedditL2 data set out of domain"""

    def __init__(self, _):
        super(RedditOutOfDomainDataProcessor, self).__init__(_)
        self.europe_user2examples = {}
        self.non_europe_user2examples = {}
        self.europe_usernames = set()
        self.non_europe_usernames = set()
        self.lang2usernames = defaultdict(list)
        self.indomain_users = set()
        self.out_of_domain_users = set()

    def fill_users(self, data_dir: str) -> set:
        suffix = '/europe_data'
        for language_folder in os.listdir(data_dir + suffix):
            for username in os.listdir(f'{data_dir}/{suffix}/{language_folder}'):
                self.europe_usernames.add(username)

        suffix = '/non_europe_data'
        for language_folder in os.listdir(data_dir + suffix):
            for username in os.listdir(f'{data_dir}/{suffix}/{language_folder}'):
                self.non_europe_usernames.add(username)


    def discover_examples(self, data_dir: str, is_europe=True)-> None:
        for language_folder in os.listdir(data_dir):
            language = language_folder
            for username in os.listdir(f'{data_dir}/{language_folder}'):
                self.lang2usernames[language].append(username)
                user_examples = []
                for chunk in os.listdir(f'{data_dir}/{language_folder}/{username}'):
                    full_path = f'{data_dir}/{language_folder}/{username}/{chunk}'
                    with open(full_path, 'r') as f:
                        sub_chunks = split_text_chunk_lines(f.readlines())

                        for i, sub_chunk in enumerate(sub_chunks):
                            user_examples.append(
                                InputExample(guid=f'{username}_{chunk}_{i}', text_a=sub_chunk, label=language)
                            )

                if is_europe:
                    self.europe_user2examples[username] = user_examples
                else:
                    self.non_europe_user2examples[username] = user_examples

    def get_train_examples(self, data_dir: str='./data/RedditL2/reddit_downsampled') -> List[InputExample]:
        self.discover_examples(data_dir + '/europe_data')
        self.discover_examples(data_dir + '/non_europe_data', is_europe=False)
        self.fill_users(data_dir)

        for usernames in self.lang2usernames.values():
            num_out_of_domain_users = int(len(usernames) * 0.1) # 10 percent of the users for testing
            usernames = set(usernames)

            out_of_domain_overlap = usernames.intersection(self.non_europe_usernames)
            non_europe_users = set(random.sample(out_of_domain_overlap, num_out_of_domain_users))
            self.out_of_domain_users = self.out_of_domain_users.union(non_europe_users)

            # Use the ramaining possible users for training
            self.indomain_users = self.indomain_users.union(usernames.difference(non_europe_users))

        examples = []
        for username in self.indomain_users:
            for example in self.europe_user2examples[username]:
                examples.append(example)

        return examples

    def get_dev_examples(self, data_dir: str=''):
        # Assumes get_training_examples have already been run
        examples = []
        for username in self.out_of_domain_users:
            for example in self.non_europe_user2examples[username]:
                examples.append(example)

        return examples

class AllOfRedditDataProcessor(RedditInDomainDataProcessor):
    def __init__(self, fold_number):
        super(AllOfRedditDataProcessor, self).__init__(fold_number)
        self.examples = []
        self.k_fold = KFold(n_splits=10)

    def merge_domains(self, europe_examples, non_europe_examples):
        ''' Do a simple random combination of the two domains for now '''
        all_examples = europe_examples + non_europe_examples
        return random.sample(all_examples, len(all_examples))

    def get_train_examples(self, data_dir: str='./data/RedditL2/text_chunks') -> List[InputExample]:
        europe_examples = self.discover_examples(data_dir + '/europe_data')
        non_europe_examples = self.discover_examples(data_dir + '/non_europe_data', indomain=False)

        self.examples = self.merge_domains(europe_examples, non_europe_examples)

        training_idxs = [fold for fold in self.k_fold.split(self.examples)][self.fold_number][0]
        return [self.examples[i] for i in training_idxs]

    def get_dev_examples(self, data_dir: str='./data/RedditL2/text_chunks'):
        ''' Assumes get_train_examples has already been run '''
        dev_idxs = [fold for fold in self.k_fold.split(self.examples)][self.fold_number][1]
        return [self.examples[i] for i in dev_idxs]

    def discover_examples(self, data_dir: str, indomain=True):
        examples = []
        for language_folder in os.listdir(data_dir):
            if language_folder.split('.')[1] == 'Ukraine':
                continue

            language = self.label2language[language_folder.split('.')[1]]

            for username in os.listdir(f'{data_dir}/{language_folder}'):
                for chunk in os.listdir(f'{data_dir}/{language_folder}/{username}'):
                    full_path = f'{data_dir}/{language_folder}/{username}/{chunk}' 
                    with open(full_path, 'r') as f:
                        sub_chunks = split_text_chunk_lines(f.readlines())

                        prefix = '' if indomain else 'out_of_dom_'
                        for i, sub_chunk in enumerate(sub_chunks):
                            examples.append(
                                InputExample(guid=f'{prefix}{username}_{chunk}_{i}', text_a=sub_chunk, label=language)
                            )
        return examples

class RedditOutOfDomainToInDomainDataProcessor(AllOfRedditDataProcessor):
    def __init__(self, _):
        super(RedditOutOfDomainToInDomainDataProcessor, self).__init__(_)
        self.europe_examples = []
        self.non_europe_examples = []

    def get_train_examples(self, data_dir: str='./data/RedditL2/text_chunks') -> List[InputExample]:
        self.europe_examples = self.discover_examples(data_dir + '/europe_data')
        self.non_europe_examples = self.discover_examples(data_dir + '/non_europe_data', indomain=False)

        return self.non_europe_examples

    def get_dev_examples(self, data_dir: str='./data/RedditL2/text_chunks'):
        ''' Assumes get_train_examples has already been run '''
        return self.europe_examples
