import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, normalize=True):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cmap = sns.cm.rocket_r

    return sns.heatmap(cm, xticklabels=classes, yticklabels=classes, annot=True, cmap=cmap, fmt='.2f')
