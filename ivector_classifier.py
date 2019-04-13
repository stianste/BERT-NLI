import numpy as np
import pandas as pd
import json
import logging

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def load_ivectors(filepath):
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
        return data

def get_id2label(filepath):
    id2label = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            example_id, _, _, label = line.split(',')
            id2label[example_id.strip()] = label.strip()

    return id2label

def get_examples_and_labels_from_map(data_map, id2label):
    data = [None for x in range(len(data_map))] 
    labels = [None for x in range(len(data_map))] 
    guids = [None for x in range(len(data_map))]

    for i, guid in enumerate(sorted(data_map.keys())):
        data[i] = np.array(data_map[guid])
        labels[i] = id2label[guid]
        guids[i] = guid

    return data, labels, guids

def save_csv(guids, y, outputs, probabilities, classes, filename):
    data_dict = {
        "guid" : guids,
        "input_label" : y,
        "output_label" : outputs,
    }
    for i in range(len(classes)):
        data_dict[classes[i]] = probabilities[:,i] # get the i-th class column

    prediction_df = pd.DataFrame(data_dict)

    prediction_df.to_csv(f'./common_predictions/{filename}.csv', index=False)

def main():
    training_path = './data/NLI-shared-task-2017/data/features/ivectors/train/ivectors.json'
    training_labels_path = './data/NLI-shared-task-2017/data/labels/train/labels.train.csv'
    dev_path = './data/NLI-shared-task-2017/data/features/ivectors/dev/ivectors.json'
    dev_labels_path = './data/NLI-shared-task-2017/data/labels/dev/labels.dev.csv'

    logger.info('Loading vectors and labels...')
    training_data_map = load_ivectors(training_path)
    training_id2labels = get_id2label(training_labels_path)

    logger.info('Merging vectors and labels...')
    X, y, training_guids = get_examples_and_labels_from_map(training_data_map, training_id2labels)

    logger.info('Training classifier')
    model = MLPClassifier(verbose=True)
    model.fit(X, y)
    classes = model.classes_

    training_outputs = model.predict(X)
    training_probas = model.predict_proba(X)

    save_csv(training_guids, y, training_outputs, training_probas, classes, 'train-ivec')

    logger.info('Loading dev data')
    dev_data_map = load_ivectors(dev_path)

    dev_data_map = load_ivectors(dev_path)
    dev_id2labels = get_id2label(dev_labels_path)

    logger.info('Evaluating model')
    X, y, dev_guids = get_examples_and_labels_from_map(dev_data_map, dev_id2labels)

    outputs = model.predict(X)
    probabilities = model.predict_proba(X)

    save_csv(dev_guids, y, outputs, probabilities, classes, 'dev-ivec')

    eval_accuracy = model.score(X, y)
    logger.info(f'Accuracy: {eval_accuracy:.3f}%')

if __name__ == '__main__':
    main()