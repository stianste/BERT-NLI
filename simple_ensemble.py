import logging
import os
import pandas as pd

from math import exp
from scipy.special import softmax

from data_processors import TOEFL11Processor, RedditInDomainDataProcessor

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from xgboost import XGBClassifier

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class WordStemTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.porter = PorterStemmer()

    def transform(self, X):
        X_trans = []
        for example in X:
            words = word_tokenize(example)
            words = list(map(self.porter.stem, words))

            X_trans.append(' '.join(words))

        return X_trans

    def fit(self, X, y=None):
        return self

class FunctionWordTransformer(WordStemTransformer):
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    def transform(self, X):
        X_trans = []
        for example in X:
            words = word_tokenize(example)
            function_words = list(filter(lambda word: word in self.stopwords, words))
            X_trans.append(' '.join(function_words))

        return X_trans

def get_prediction_data(dir_path, name, model_type, max_features):
    matches = list(filter(lambda filename: filename.startswith(f'{name}_{model_type}_{max_features}'), 
                                os.listdir(dir_path)))

    if len(matches) > 0:
        return pd.read_csv(dir_path + matches[0]).sort_values(by=['guid'])

    return pd.DataFrame()

def str2model(model_name):
    models = {
        'svm' : SVC(kernel='linear', cache_size=4098, decision_function_shape='ovr', probability=True),
        'ffnn' : MLPClassifier(),
        'XGBoost' : XGBClassifier(max_depth=20),
        'lda' : LinearDiscriminantAnalysis(),
    }
    return models[model_name]

def get_tfidf_pipeline_for_model(model_name, ngram_range, analyzer, max_features):

    return [
        ('tf-idf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, analyzer=analyzer)),
        (model_name, str2model(model_name))
    ]

def get_lemma_pipeline_for_model(model_name, ngram_range, analyzer, max_features):
    return [ ('lemma', WordStemTransformer())] + get_tfidf_pipeline_for_model(model_name, ngram_range, analyzer, max_features)

def get_func_word_pipeline_for_model(model_name, ngram_range, analyzer, max_features):
    return [ ('function-words', FunctionWordTransformer())] + get_tfidf_pipeline_for_model(model_name, ngram_range, analyzer, max_features)

def get_toefl_data():
    data_proc = TOEFL11Processor()
    examples = data_proc.get_train_examples()
    training_examples = [(ex.guid, ex.text_a) for ex in examples]
    y_train = [(ex.guid, ex.label) for ex in examples]

    examples = data_proc.get_dev_examples()
    test_examples = [(ex.guid, ex.text_a) for ex in examples]
    y_test = [(ex.guid, ex.label) for ex in examples]

    return training_examples, y_train, test_examples, y_test

def save_results(predictions_path, model_name, bagging_estimator, estimators, max_features, with_bert, eval_acc, f1):
    base_estimator_name = estimators[0][1].steps[-1][0]
    bert_string = 'wBERT_' if with_bert else '' 

    filename = f'{model_name}_{bagging_estimator}_{base_estimator_name}_{max_features}_{bert_string}{eval_acc:.3f}_{f1:.3f}'
    with open(predictions_path + f'/results/{filename}.txt', 'w') as f:
        f.write(f'accuracy : {eval_acc}\n')
        f.write(f'f1 : {f1}\n')
        f.write(', '.join([estimator[0] for estimator in estimators]))

def merge_with_bert(df, predictions_path, scenario, bert_output_type=None):
    dir_path = f'{predictions_path}/{scenario}'
    bert_filename = list(filter(lambda filename: filename.startswith('bert'),
                                os.listdir(dir_path)))[0]

    def map_logit_to_probability(logit):
        odds = exp(logit)
        prob = odds / (1 + odds)
        return prob

    bert_df = pd.read_csv(f'{dir_path}/{bert_filename}').drop(columns=['input', 'output', 'input_label', 'output_label'])
    bert_df = bert_df.sort_values(by=['guid'])

    non_guid_columns = bert_df.columns.difference(['guid'])
    if bert_output_type == 'probabilities':
        # Map logits to probabilities for all columns, except guid
        bert_df[non_guid_columns] = bert_df[non_guid_columns].applymap(lambda cell: map_logit_to_probability(cell))

    elif bert_output_type == 'softmax':
        bert_df[non_guid_columns] = bert_df[non_guid_columns].apply(lambda row: softmax(row), axis=1)

    elif bert_output_type == 'all':
        bert_df[[col + '_prop' for col in non_guid_columns]] = bert_df[non_guid_columns].applymap(lambda cell: map_logit_to_probability(cell))
        bert_df[[col + '_softmax' for col in non_guid_columns]] = bert_df[non_guid_columns].apply(lambda row: softmax(row), axis=1)

    combined_df = pd.merge(df, bert_df, on=['guid'])
    return combined_df

def main():
    reddit = False
    bert_output_type = ''

    num_bagging_classifiers = 200
    max_samples = 0.8
    mem_path = './common_predictions/cache'
    prefix = 'reddit' if reddit else 'toefl'
    predictions_path = f'./common_predictions/{prefix}_predictions'
    
    training_examples, y_train, test_examples, y_test = get_toefl_data()

    training_guids, training_examples_no_guid = zip(*training_examples)
    y_train_guids, y_train_no_guid = zip(*y_train)
    test_guids, test_examples_no_guid = zip(*test_examples)
    y_test_guids, y_test_no_guid = zip(*y_test)

    base_model_type = 'ffnn'
    stack_type = 'meta_classifier'
    meta_classifier_type = 'lda'
    max_features = None
    use_bert = True

    logger.info(f'Running {max_features} {stack_type} {meta_classifier_type} {base_model_type}')
    char_2_gram_pipeline = Pipeline(get_tfidf_pipeline_for_model(base_model_type, (2,2), 'char', max_features), memory=mem_path)
    char_3_gram_pipeline = Pipeline(get_tfidf_pipeline_for_model(base_model_type, (3,3), 'char', max_features), memory=mem_path)
    char_4_gram_pipeline = Pipeline(get_tfidf_pipeline_for_model(base_model_type, (4,4), 'char', max_features), memory=mem_path)

    word_1_gram_pipeline = Pipeline(get_tfidf_pipeline_for_model(base_model_type, (1,1), 'word', max_features), memory=mem_path)
    word_2_gram_pipeline = Pipeline(get_tfidf_pipeline_for_model(base_model_type, (2,2), 'word', max_features), memory=mem_path)
    word_3_gram_pipeline = Pipeline(get_tfidf_pipeline_for_model(base_model_type, (3,3), 'word', max_features), memory=mem_path)

    lemma_1_gram_pipeline = Pipeline(get_lemma_pipeline_for_model(base_model_type, (1,1), 'word', max_features), memory=mem_path)
    lemma_2_gram_pipeline = Pipeline(get_lemma_pipeline_for_model(base_model_type, (2,2), 'word', max_features), memory=mem_path)

    func_1_gram_pipeline = Pipeline(get_func_word_pipeline_for_model(base_model_type, (1,1), 'word', max_features), memory=mem_path)
    func_2_gram_pipeline = Pipeline(get_func_word_pipeline_for_model(base_model_type, (2,2), 'word', max_features), memory=mem_path)

    estimators = [
        ('char2', char_2_gram_pipeline),
        ('char3', char_3_gram_pipeline),
        ('char4', char_4_gram_pipeline),

        ('word1', word_1_gram_pipeline),
        ('word2', word_2_gram_pipeline),
        ('word3', word_3_gram_pipeline),

        ('lemma1', lemma_1_gram_pipeline),
        ('lemma2', lemma_2_gram_pipeline),

        ('func1', func_1_gram_pipeline),
        ('func2', func_2_gram_pipeline),
    ]
    for name, pipeline in estimators:
        model_name = pipeline.steps[-1][0]
        if not get_prediction_data(predictions_path + '/train/', name, model_name, max_features).empty:
            logger.info(f'Skipping {name} {model_name} {max_features}')
            continue

        logger.info(f'Traning {name} {model_name} {max_features}...')

        pipeline.fit(training_examples_no_guid, y_train_no_guid)

        training_predictions = pipeline.predict_proba(training_examples_no_guid)
        test_predictions = pipeline.predict_proba(test_examples_no_guid)

        classes = pipeline.steps[-1][1].classes_

        training_df = pd.DataFrame(data=training_predictions, columns=classes)
        training_df['guid'] = training_guids
        training_df['y_guid'] = y_train_guids

        test_df = pd.DataFrame(data=test_predictions, columns=classes)
        test_df['guid'] = test_guids
        test_df['y_guid'] = y_test_guids

        eval_acc = pipeline.score(test_examples_no_guid, y_test_no_guid)

        logger.info(f'Model accuracy: {eval_acc}')

        training_df.to_csv(f'{predictions_path}/train/{name}_{model_name}_{max_features}_{eval_acc:.3f}.csv', index=False)
        test_df.to_csv(f'{predictions_path}/test/{name}_{model_name}_{max_features}_{eval_acc:.3f}.csv', index=False)

    training_frames = []
    test_frames = []
    for name, pipeline in estimators:
        model_name = pipeline.steps[-1][0]

        training_df = get_prediction_data(predictions_path + '/train/', name, model_name, max_features)
        test_df = get_prediction_data(predictions_path + '/test/', name, model_name, max_features)

        training_frames.append(training_df)
        test_frames.append(test_df)

    # all_training_data_df = pd.concat(training_frames, axis=1)
    all_training_data_df = training_frames[0]
    all_test_data_df = test_frames[0]
    for i in range(1, len(training_frames)):
        all_training_data_df = pd.merge(all_training_data_df, training_frames[i], on='guid')
        all_test_data_df = pd.merge(all_test_data_df, test_frames[i], on='guid')

    if use_bert:
        logger.info('Merging with BERT')
        all_training_data_df = merge_with_bert(all_training_data_df, predictions_path, 'train', bert_output_type)
        all_test_data_df = merge_with_bert(all_test_data_df, predictions_path, 'test', bert_output_type)

    all_training_data_df.to_csv('./common_predictions/all_training_data.csv', index=False)

    drop_columns = [column_name for column_name in all_training_data_df.columns if 'guid' in column_name]
    logger.info(f'Dropping columns: {drop_columns}')
    all_training_data = all_training_data_df.drop(columns=drop_columns).to_numpy()
    all_test_data = all_test_data_df.drop(columns=drop_columns).to_numpy()

    if stack_type == 'meta_classifier':
        model = str2model(meta_classifier_type)
        bagging_estimator = ''
    else:
        base_estimator = str2model(meta_classifier_type)
        model = BaggingClassifier(base_estimator,
                                n_estimators=num_bagging_classifiers, max_samples=max_samples)
        # model = AdaBoostClassifier(base_estimator=base_estimator)
        # model = GradientBoostingClassifier(base_estimator)
        bagging_estimator = type(base_estimator).__name__

    logger.info(f'All training data shape: {all_training_data.shape}')
    logger.info(f'All test data shape: {all_test_data.shape}')
    logger.info(f'First labels: {y_train[:10]}')
    logger.info(f'Last labels: {y_train[-10:]}')

    model.fit(all_training_data, y_train_no_guid)

    eval_predictions = model.predict(all_test_data)
    eval_acc = model.score(all_test_data, y_test_no_guid)

    macro_f1 = f1_score(y_test_no_guid, eval_predictions, average='macro')
    logger.info(f'Final {stack_type} eval accuracy {eval_acc}. F1: {macro_f1}')

    model_name = type(model).__name__

    save_results(predictions_path, model_name, bagging_estimator, estimators, max_features, use_bert, eval_acc, macro_f1)

if __name__ == '__main__':
    main()
