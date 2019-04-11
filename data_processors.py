import constants
import os
import random
from collections import defaultdict

from typing import List

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

    def __init__(self, use_reddit_labels: bool=False):
        self.id2label = {}
        self.use_reddit_labels = use_reddit_labels
        self.label_white_list = None
        if self.use_reddit_labels:
            self.label_map = {
                "HIN" : "Hindi",
                "ARA" : "Arabic",
                "JPN" : "Japanese",
                "GER" : "German",
                "TEL" : "Telugu",
                "KOR" : "Korean",
                "SPA" : "Spanish",
                "ITA" : "Italian",
                "CHI" : "Chinese",
                "FRE" : "French",
                "TUR" : "Turkish",
            }

    def set_label_white_list(self, label_white_list: set):
        self.label_white_list = label_white_list

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        full_path = data_dir + constants.TOEFL11_TRAINING_DATA_PATH
        self._create_labels(data_dir)
        return self._get_examples(full_path)

    def get_dev_examples(self, data_dir):
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

        if self.use_reddit_labels:
            return [self.label_map[label] for label in labels]

        return labels

    def _get_examples(self, full_path: str) -> List[InputExample]:
        examples = []

        for filename in os.listdir(full_path):
            example_id = filename.split('.')[0]
            with open(os.path.join(full_path, filename), "r") as f:
                label = self.id2label[example_id]

                if self.label_white_list and label not in self.label_white_list:
                    continue

                text = "".join(f.readlines()).lower()
                example_id = filename.split(".")[0]

                if self.use_reddit_labels:
                    label = self.label_map[label]

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
                if self.use_reddit_labels:
                    label = self.label_map[label.strip()]

                if self.label_white_list and label not in self.label_white_list:
                    continue

                self.id2label[example_id.strip()] = label.strip()


class RedditInDomainDataProcessor(DataProcessor):
    """Processor for the RedditL2 data set"""

    def __init__(self, fold_number):
        self.fold_number = fold_number - 1
        self.user2examples = {}
        self.lang2usernames = defaultdict(list)

    def discover_examples(self, data_dir: str)-> None:
        for language_folder in os.listdir(data_dir):
            language = language_folder
            for username in os.listdir(f'{data_dir}/{language_folder}'):
                self.lang2usernames[language].append(username)
                user_examples = []
                for chunk in os.listdir(f'{data_dir}/{language_folder}/{username}'):
                    full_path = f'{data_dir}/{language_folder}/{username}/{chunk}' 
                    with open(full_path, 'r') as f:
                        text = ''.join(f.readlines()).lower()

                        user_examples.append(
                            InputExample(guid=f'{username}_{chunk}', text_a=text, label=language)
                        )

                self.user2examples[username] = user_examples

        user_has_more_than_one_chunk = False

        # Make sure everything is in order
        for username, example_list in self.user2examples.items():
            num_examples = len(example_list)

            if num_examples > 1:
                user_has_more_than_one_chunk = True

        if not user_has_more_than_one_chunk:
            print('All users have only one example/chunk')
             
        for language, user_list in self.lang2usernames.items():
            assert len(user_list) == 104, f'{language} has {len(user_list)} users.'

    def _get_examples_for_fold(self, fold_function):
        examples = []
        fold_size = int(104 / 10)

        for usernames in self.lang2usernames.values():
            usernames_for_lang = fold_function(usernames, self.fold_number, fold_size)
            for username in usernames_for_lang:
                for example in self.user2examples[username]:
                    examples.append(example)

        return examples

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        self.discover_examples(data_dir)
        return self._get_examples_for_fold(self._get_train_fold)

    def get_dev_examples(self, data_dir):
        return self._get_examples_for_fold(self._get_dev_fold)

    def get_labels(self):
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
            assert label in label2language

        language_list = list(set(label2language.values()))
        assert len(language_list) == 23
        return language_list 

class RedditOutOfDomainDataProcessor(RedditInDomainDataProcessor):
    """Processor for the RedditL2 data set out of domain"""

    def __init__(self, _):
        self.europe_user2examples = {}
        self.non_europe_user2examples = {}
        self.europe_usernames = set()
        self.non_europe_usernames = set()
        self.lang2usernames = defaultdict(list)

    def verify_users(self, data_dir: str) -> set:
        europe_users = set()
        for language_folder in os.listdir(data_dir + '/europe_data'):
            for username in os.listdir(f'{data_dir}/{language_folder}'):
                europe_users.add(username)

        non_europe_users = set()
        for language_folder in os.listdir(data_dir + '/non_europe_data'):
            for username in os.listdir(f'{data_dir}/{language_folder}'):
                non_europe_users.add(username)


    def discover_examples(self, data_dir: str, is_europe=True)-> None:
        for language_folder in os.listdir(data_dir):
            language = language_folder
            for username in os.listdir(f'{data_dir}/{language_folder}'):
                self.lang2usernames[language].append(username)
                user_examples = []
                for chunk in os.listdir(f'{data_dir}/{language_folder}/{username}'):
                    full_path = f'{data_dir}/{language_folder}/{username}/{chunk}'
                    with open(full_path, 'r') as f:
                        text = ''.join(f.readlines()).lower()

                        user_examples.append(
                            InputExample(guid=f'{username}_{chunk}', text_a=text, label=language)
                        )

                if is_europe:
                    self.europe_user2examples[username] = user_examples
                else:
                    self.non_europe_user2examples[username] = user_examples

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        self.discover_examples(data_dir + '/europe_data')
        self.discover_examples(data_dir + '/non_europe_data', is_europe=False)
        self.verify_users(data_dir)

        for usernames in self.lang2usernames.values():
            num_europe_users = int(len(usernames) * 0.9)
            europe_users = random.sample(list(usernames), num_europe_users)
            self.europe_usernames = self.europe_usernames.union(europe_users)
            self.non_europe_usernames = self.non_europe_usernames.union(usernames.difference(europe_users))

        examples = [None for x in range(len(self.europe_usernames))]
        for i, username in enumerate(self.europe_usernames):
            for example in self.europe_user2examples[username]:
                examples[i] = example

        return examples

    def get_dev_examples(self, data_dir):
        # Assumes get_training_examples have already been run
        examples = [None for x in range(len(self.non_europe_usernames))]
        for i, username in enumerate(self.non_europe_usernames):
            for example in self.non_europe_user2examples[username]:
                examples[i] = example

        return examples

class CommonLabelsReddit2TOEFL11Processor(DataProcessor):
    """
    A large data processor which uses all the data available and tests on the toefl dev set.
    """
    def __init__(self):
        self.toefl_processor = TOEFL11Processor()
        self.reddit_processor = RedditInDomainDataProcessor(0)

        self.reddit_labels_to_toefl = {
            "Germany" : "GER",
            "Spain" : "SPA",
            "Italy" : "ITA",
            "France" : "FRE",
            "Turkey" : "TUR",
        }

    def get_train_examples(self, data_dir):
        toefl_data_dir = './data/NLI-shared-task-2017/' + constants.TOEFL11_TRAINING_DATA_PATH
        examples = self.toefl_processor.get_train_examples(toefl_data_dir)

        # Reddit Europe data
        examples += self._get_reddit_examples(data_dir + '/europe_data')
        examples += self._get_reddit_examples(data_dir + '/non_europe_data')

        return examples

    def _get_reddit_examples(self, data_dir):
        examples = []
        for language_folder in os.listdir(data_dir):
            language = language_folder.split('.')[0]

            if not language in self.reddit_labels_to_toefl:
                continue

            language = self.reddit_labels_to_toefl[language]

            for username in os.listdir(f'{data_dir}/{language_folder}'):
                for chunk in os.listdir(f'{data_dir}/{language_folder}/{username}'):
                    full_path = f'{data_dir}/{language_folder}/{username}/{chunk}'
                    with open(full_path, 'r') as f:
                        text = ''.join(f.readlines()).lower()
                        examples.append(
                            InputExample(guid=f'{username}_{chunk}', text_a=text, label=language)
                        )

        return examples


    def get_dev_examples(self, _):
        data_dir = './data/NLI-shared-task-2017/' + constants.TOEFL11_DEV_DATA_PATH
        return self.toefl_processor.get_dev_examples(data_dir)

    def get_labels(self):
        return self.toefl_processor.get_labels()

class LargeReddit2TOEFL11Processor(DataProcessor):
    """
    A large data processor which uses all the data available and tests on the toefl dev set.
    """
    def __init__(self):
        self.toefl_processor = TOEFL11Processor(use_reddit_labels=True)
        self.reddit_processor = RedditInDomainDataProcessor(0)
        self.get_labels()
        self.toefl_processor.set_label_white_list(self.common_labels)

    def get_train_examples(self, data_dir):
        examples = []
        europe_dir = data_dir + '/europe_data'
        for language_folder in os.listdir(europe_dir):
            language = language_folder
            if language not in self.common_labels:
                continue

            for username in os.listdir(f'{europe_dir}/{language_folder}'):
                for chunk in os.listdir(f'{europe_dir}/{language_folder}/{username}'):
                    full_path = f'{europe_dir}/{language_folder}/{username}/{chunk}'
                    with open(full_path, 'r') as f:
                        text = ''.join(f.readlines()).lower()
                        examples.append(
                            InputExample(guid=f'{username}_{chunk}', text_a=text, label=language)
                        )

        non_europe_dir = data_dir + '/non_europe_data'
        for language_folder in os.listdir(non_europe_dir):
            language = language_folder
            if language not in self.common_labels:
                continue

            for username in os.listdir(f'{non_europe_dir}/{language_folder}'):
                for chunk in os.listdir(f'{non_europe_dir}/{language_folder}/{username}'):
                    full_path = f'{non_europe_dir}/{language_folder}/{username}/{chunk}'
                    with open(full_path, 'r') as f:
                        text = ''.join(f.readlines()).lower()

                        examples.append(
                            InputExample(guid=f'{username}_{chunk}', text_a=text, label=language)
                        )

        toefl_data_dir = './data/NLI-shared-task-2017/' + constants.TOEFL11_TRAINING_DATA_PATH
        examples += self.toefl_processor.get_train_examples(toefl_data_dir)

        return examples

    def get_dev_examples(self, _):
        data_dir = './data/NLI-shared-task-2017/' + constants.TOEFL11_DEV_DATA_PATH
        return self.toefl_processor.get_dev_examples(data_dir)

    def get_labels(self):
        toefl_labels = self.toefl_processor.get_labels()
        reddit_labels = self.reddit_processor.get_labels()
        self.common_labels = set(toefl_labels).intersection(reddit_labels)
        return list(self.common_labels)
