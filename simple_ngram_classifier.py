from data_processors import TOEFL11Processor, RedditInDomainDataProcessor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer

max_features = None
ngram_range = (1,1)
dec_func_shape = 'ovr'
analyzer = 'word' # Can also be 'word'

reddit = True

if not reddit:
    data_proc = TOEFL11Processor()
    training_examples = [ex.text_a for ex in data_proc.get_train_examples()]
    y_train = [ex.label for ex in data_proc.get_train_examples()]

    test_examples = [ex.text_a for ex in data_proc.get_dev_examples()]
    y_test = [ex.label for ex in data_proc.get_dev_examples()]

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, analyzer=analyzer)
    normalized_examples = vectorizer.fit_transform(training_examples + test_examples)
    X_train = normalized_examples[:11000]
    X_test = normalized_examples[11000:]

    # model = SVC(verbose=True, kernel='linear', cache_size=2048, decision_function_shape=dec_func_shape)
    # model = MLPClassifier(verbose=True)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    eval_acc = model.score(X_test, y_test)
    print('Accuracy:', eval_acc)

else:
    accuracies = []

    for k_fold in range(1, 11):
        data_proc = RedditInDomainDataProcessor(k_fold)
        training_input_examples = data_proc.get_train_examples()
        training_examples = [ex.text_a for ex in training_input_examples]
        y_train = [ex.label for ex in training_input_examples]

        test_examples = [ex.text_a for ex in data_proc.get_dev_examples()]
        y_test = [ex.label for ex in data_proc.get_dev_examples()]

        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, analyzer=analyzer)
        normalized_examples = vectorizer.fit_transform(training_examples + test_examples)
        X_train = normalized_examples[:len(training_examples)]
        X_test = normalized_examples[len(training_examples):]

        # model = SVC(verbose=True, kernel='linear', cache_size=2048, decision_function_shape=dec_func_shape)
        # model = MLPClassifier(verbose=True)
        model = MultinomialNB()
        model.fit(X_train, y_train)

        eval_acc = model.score(X_test, y_test)
        print('Accuracy:', eval_acc)

        accuracies.append(eval_acc)

    print('Average over 10 folds', sum(accuracies)/len(accuracies))
