import logging
import os
import pandas as pd

from data_processors import TOEFL11Processor, RedditInDomainDataProcessor

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class WordStemTransformer(TransformerMixin):
    def __init__(self):
        self.porter = PorterStemmer()

    def transform(self, X):
        X_trans = []
        for example in X:
            words = word_tokenize(example)
            words = map(self.porter.stem, words)
            X_trans.append(''.join(words))

        return X_trans

    def fit(self, X, y=None):
        return self

def get_prediction_data(dir_path, name, model_type, max_features):
    matches = list(filter(lambda filename: filename.startswith(f'{name}_{model_type}_{max_features}'), 
                                os.listdir(dir_path)))

    if len(matches) > 0:
        return pd.read_csv(dir_path + matches[0])

    return pd.DataFrame()

def get_tfidf_svm_with_arguments(ngram_range, analyzer, max_features, dec_func_shape='ovr'):
    return [
        ('tf-idf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, analyzer=analyzer)),
        # ('svm', SVC(kernel='linear', cache_size=2048, decision_function_shape=dec_func_shape, probability=True))
        ('ffnn', MLPClassifier())
    ]

def get_toefl_data():
    data_proc = TOEFL11Processor()
    training_examples = [ex.text_a for ex in data_proc.get_train_examples()]
    y_train = [ex.label for ex in data_proc.get_train_examples()]

    test_examples = [ex.text_a for ex in data_proc.get_dev_examples()]
    y_test = [ex.label for ex in data_proc.get_dev_examples()]

    return training_examples, y_train, test_examples, y_test

def main():
    max_features = 10000
    stack_type = '' # 'simple_ensemble', 'meta_classifier', 'meta_ensemble'
    mem_path = './common_predictions/cache'
    predictions_path = './common_predictions/predictions'

    char_1_gram_pipeline = Pipeline(get_tfidf_svm_with_arguments((1,1), 'char', max_features), memory=mem_path)
    char_2_gram_pipeline = Pipeline(get_tfidf_svm_with_arguments((2,2), 'char', max_features), memory=mem_path)
    char_3_gram_pipeline = Pipeline(get_tfidf_svm_with_arguments((3,3), 'char', max_features), memory=mem_path)

    word_1_gram_pipeline = Pipeline(get_tfidf_svm_with_arguments((1,1), 'word', max_features), memory=mem_path)
    word_2_gram_pipeline = Pipeline(get_tfidf_svm_with_arguments((2,2), 'word', max_features), memory=mem_path)
    word_3_gram_pipeline = Pipeline(get_tfidf_svm_with_arguments((3,3), 'word', max_features), memory=mem_path)

    lemma_2_gram_pipeline = Pipeline(
            [('stem', WordStemTransformer())] + 
            get_tfidf_svm_with_arguments((2,2),'word', max_features), memory=mem_path
        )

    estimators = [
        ('char1', char_1_gram_pipeline),
        ('char2', char_2_gram_pipeline),
        ('char3', char_3_gram_pipeline),

        ('word1', word_1_gram_pipeline),
        ('word2', word_2_gram_pipeline),
        ('word3', word_3_gram_pipeline),
        ('lemma', lemma_2_gram_pipeline),
    ]
    

    training_examples, y_train, test_examples, y_test = get_toefl_data()

    logger.info(f'Stack type: {stack_type}')
    if stack_type == 'simple_ensemble':
        ensemble_classifier = VotingClassifier(estimators=estimators, voting='soft', n_jobs=2)
        ensemble_classifier.fit(training_examples, y_train)
        eval_acc = ensemble_classifier.score(test_examples, y_test)
        logger.info(f'Final eval accuracy {eval_acc}')
        exit()

    for name, pipeline in estimators:
        model_name = pipeline.steps[-1][0]
        if not get_prediction_data(predictions_path + '/train/', name, model_name, max_features).empty:
            logger.info(f'Skipping {name} {model_name} {max_features}')
            continue

        logger.info(f'Name: {name}')
        pipeline.fit(training_examples, y_train)

        training_predictions = pipeline.predict_proba(training_examples)
        test_predictions = pipeline.predict_proba(test_examples)

        classes = pipeline.steps[-1][1].classes_

        training_df = pd.DataFrame(data=training_predictions, columns=classes)
        test_df = pd.DataFrame(data=test_predictions, columns=classes)

        eval_acc = pipeline.score(test_examples, y_test)
        logger.info(f'Model accuracy: {eval_acc}')

        training_df.to_csv(f'{predictions_path}/train/{name}_{model_name}_{max_features}_{eval_acc:.3f}.csv', index=False)
        test_df.to_csv(f'{predictions_path}/test/{name}_{model_name}_{max_features}_{eval_acc:.3f}.csv', index=False)

    training_frames = []
    test_frames = []
    for name, pipeline in estimators:
        model_name = pipeline.steps[-1][0]
        training_df = get_prediction_data(predictions_path + '/train/', name, model_name, max_features)
        training_frames.append(training_df)
        test_df = get_prediction_data(predictions_path + '/test/', name, model_name, max_features)
        test_frames.append(test_df)

    all_training_data = pd.concat(training_frames, axis=1).values
    all_test_data = pd.concat(test_frames, axis=1).values

    if stack_type == 'meta_classifier':
        model = LinearDiscriminantAnalysis()

    else:
        model = BaggingClassifier(LinearDiscriminantAnalysis())

    model.fit(all_training_data, y_train)
    eval_acc = model.score(all_test_data, y_test)
    logger.info(f'Final {stack_type} eval accuracy {eval_acc}')


if __name__ == '__main__':
    main()