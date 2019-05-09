from data_processors import TOEFL11Processor, RedditInDomainDataProcessor
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier


from sklearn.feature_extraction.text import TfidfVectorizer

max_features = 10000
dec_func_shape = 'ovr'

data_proc = TOEFL11Processor()
training_examples = [ex.text_a for ex in data_proc.get_train_examples()]
y_train = [ex.label for ex in data_proc.get_train_examples()]

test_examples = [ex.text_a for ex in data_proc.get_dev_examples()]
y_test = [ex.label for ex in data_proc.get_dev_examples()]

char_1_gram_pipeline = Pipeline([
    ('tf-idf', TfidfVectorizer(max_features=max_features, ngram_range=(1,1), analyzer='char')),
    ('svm', SVC(kernel='linear', cache_size=2048, decision_function_shape=dec_func_shape, probability=True))
])

word_1_gram_pipeline = Pipeline([
    ('tf-idf', TfidfVectorizer(max_features=max_features, ngram_range=(1,1), analyzer='word')),
    ('svm', SVC(kernel='linear', cache_size=2048, decision_function_shape=dec_func_shape, probability=True))
])

ensemle_classifier = VotingClassifier(estimators=
    [
        ('char 1', char_1_gram_pipeline), 
        ('word 1', word_1_gram_pipeline),
    ], voting='soft', n_jobs=-1)


ensemle_classifier.fit(training_examples, y_train)
eval_acc = ensemle_classifier.score(test_examples, y_test)

print(f'Final eval accuracy {eval_acc}')