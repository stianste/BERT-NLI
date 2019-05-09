from data_processors import TOEFL11Processor, RedditInDomainDataProcessor
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_svm_pipline_with_arguments(ngram_range, analyzer, max_features, dec_func_shape='ovr'):
    return Pipeline([
        ('tf-idf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, analyzer=analyzer)),
        ('svm', SVC(kernel='linear', cache_size=2048, decision_function_shape=dec_func_shape, probability=True))
    ])

def get_toefl_data():
    data_proc = TOEFL11Processor()
    training_examples = [ex.text_a for ex in data_proc.get_train_examples()]
    y_train = [ex.label for ex in data_proc.get_train_examples()]

    test_examples = [ex.text_a for ex in data_proc.get_dev_examples()]
    y_test = [ex.label for ex in data_proc.get_dev_examples()]

    return training_examples, y_train, test_examples, y_test

def main():
    max_features = 10000

    char_1_gram_pipeline = get_tfidf_svm_pipline_with_arguments((1,1), 'char', max_features)
    char_2_gram_pipeline = get_tfidf_svm_pipline_with_arguments((2,2), 'char', max_features)
    char_3_gram_pipeline = get_tfidf_svm_pipline_with_arguments((3,3), 'char', max_features)

    word_1_gram_pipeline = get_tfidf_svm_pipline_with_arguments((1,1), 'word', max_features)
    word_2_gram_pipeline = get_tfidf_svm_pipline_with_arguments((2,2), 'word', max_features)
    word_3_gram_pipeline = get_tfidf_svm_pipline_with_arguments((3,3), 'word', max_features)

    ensemle_classifier = VotingClassifier(
        estimators = [
            ('char 1', char_1_gram_pipeline),
            ('char 2', char_2_gram_pipeline),
            ('char 3', char_3_gram_pipeline),

            ('word 1', word_1_gram_pipeline),
            ('word 2', word_2_gram_pipeline),
            ('word 3', word_3_gram_pipeline),
        ], voting='soft', n_jobs=-1)

    training_examples, y_train, test_examples, y_test = get_toefl_data()

    ensemle_classifier.fit(training_examples, y_train)
    eval_acc = ensemle_classifier.score(test_examples, y_test)

    print(f'Final eval accuracy {eval_acc}')

if __name__ == '__main__':
    main()