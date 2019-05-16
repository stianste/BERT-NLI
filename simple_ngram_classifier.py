from data_processors import TOEFL11Processor, RedditInDomainDataProcessor, RedditOutOfDomainDataProcessor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

max_features = None
ngram_range = (1,1)
dec_func_shape = 'ovr'
analyzer = 'word' # Can also be 'word'

reddit = True

def train_and_evaluate_pipeline_for_model(model, indomain=True, fold_nr=0):
    pipeline = Pipeline([
        ('tf-idf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, analyzer=analyzer)),
        model 
    ])

    pipeline.fit(training_examples, y_train)

    eval_acc = pipeline.score(test_examples, y_test)
    test_predictions = pipeline.predict(test_examples)

    suffix = '' if indomain else 'out_of_domain'
    df = pd.DataFrame(
        {
            'guid' : test_guids,
            'label' : y_test,
            'output' : test_predictions,
        }
    ).to_csv(f'./results/baselines/{model[0]}/{suffix}_fold{fold_nr}.csv', index=False)
    
    print(f'Eval accuracy {eval_acc}')
    return eval_acc

if not reddit:
    data_proc = TOEFL11Processor()
    training_examples = [ex.text_a for ex in data_proc.get_train_examples()]
    y_train = [ex.label for ex in data_proc.get_train_examples()]

    test_examples = [ex.text_a for ex in data_proc.get_dev_examples()]
    y_test = [ex.label for ex in data_proc.get_dev_examples()]

    model = ('svm', SVC(verbose=True, kernel='linear', cache_size=2048, decision_function_shape=dec_func_shape))
    # model = ('naive-bayes', MultinomialNB())

    train_and_evaluate_pipeline_for_model(model)

else:
    accuracies = []

    for k_fold in range(1, 11):
        data_proc = RedditInDomainDataProcessor(k_fold)
        # data_proc = RedditOutOfDomainDataProcessor(k_fold)

        training_input_examples = sorted(data_proc.get_train_examples(), key=lambda ex: ex.guid)
        training_examples = [ex.text_a for ex in training_input_examples]

        y_train = [ex.label for ex in training_input_examples]

        dev_examples = sorted(data_proc.get_dev_examples(), key=lambda ex: ex.guid)
        test_examples = [ex.text_a for ex in dev_examples]
        test_guids = [ex.guid for ex in dev_examples]
        y_test = [ex.label for ex in dev_examples]
        print(test_guids[:10])

        model = ('svm', SVC(kernel='linear', cache_size=2048, decision_function_shape=dec_func_shape))
        # model = ('naive-bayes', MultinomialNB())

        eval_acc = train_and_evaluate_pipeline_for_model(model, fold_nr=k_fold)

        accuracies.append(eval_acc)

    print('Average over 10 folds', f'{sum(accuracies)/len(accuracies):.3f}')
