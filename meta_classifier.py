import pandas as pd
import numpy as np

from data_processors import TOEFL11Processor
from sklearn.neural_network import MLPClassifier

def get_vectors(type_prefix):
    data_dir = './common_predictions'
    bert_df = pd.read_csv(f'{data_dir}/{type_prefix}-toefl-bert-base-5-epochs.csv')
    ivec_df = pd.read_csv(f'{data_dir}/{type_prefix}-ivec.csv')

    bert_df.sort_values(by=['guid'], inplace=True)
    ivec_df.sort_values(by=['guid'], inplace=True)

    labels = TOEFL11Processor().get_labels()

    bert_df = pd.get_dummies(bert_df, columns=['output_label'])

    bert_vectors = bert_df[['logit'] + [f'output_label_{lang}' for lang in labels]].values
    ivec_vectors = ivec_df[labels].values

    X = np.concatenate((bert_vectors, ivec_vectors), axis=1)
    y = bert_df['input_label'].values # Same as ivec after sorting

    return X, y

def main():
    X_train, y_train = get_vectors('train')
    X_dev, y_dev = get_vectors('dev')

    assert X_train.shape == (11000, 23)
    assert y_train.shape == (11000,)

    assert X_dev.shape == (1100, 23)
    assert y_dev.shape == (1100,)

    model = MLPClassifier(verbose=True)
    model.fit(X_train, y_train)
    eval_acc = model.score(X_dev, y_dev)

    print('Final eval accuracy:', eval_acc)

if __name__ == '__main__':
    main()