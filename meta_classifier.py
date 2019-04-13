import pandas as pd
import numpy as np

from data_processors import TOEFL11Processor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def get_vectors(type_prefix):
    data_dir = './common_predictions'
    bert_df = pd.read_csv(f'{data_dir}/{type_prefix}-toefl-bert-base-5-epochs.csv')
    ivec_df = pd.read_csv(f'{data_dir}/{type_prefix}-ivec.csv')

    bert_df.sort_values(by=['guid'], inplace=True)
    ivec_df.sort_values(by=['guid'], inplace=True)

    labels = TOEFL11Processor().get_labels()

    bert_df = pd.get_dummies(bert_df, columns=['output_label'])

    combined_df = pd.merge(bert_df, ivec_df, on=['guid'])
    combined_df.to_csv(f'./temp/{type_prefix}_merged.csv', index=False)

    # Keep logit, dummy variables from bert, and logits from ivec
    columns = ['logit'] + labels + [f'output_label_{lang}' for lang in labels]

    X = combined_df[columns].values
    y = combined_df['input_label_x'].values

    return X, y

def main():
    X_dev, y_dev = get_vectors('dev')
    X_train, y_train = get_vectors('train')

    assert X_train.shape == (11000, 23), f'Train shape: {X_train.shape}'
    assert y_train.shape == (11000,)

    assert X_dev.shape == (1100, 23)
    assert y_dev.shape == (1100,)

    # model = MLPClassifier(verbose=True, max_iter=500)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    eval_acc = model.score(X_dev, y_dev)

    print('Final eval accuracy:', eval_acc)

if __name__ == '__main__':
    main()