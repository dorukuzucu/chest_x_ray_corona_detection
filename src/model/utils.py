import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score


def save_confusion_matrix(confisuon_matrix,title,filename):
    # plot confusion matrix
    fig,ax = plt.subplots()
    ax.matshow(confisuon_matrix, cmap=plt.cm.Blues)
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            c = confisuon_matrix[i][j]
            ax.text(j, i, str(c), va='center', ha='center')
            time.sleep(0.5)
    fig.savefig(filename+'.png')

def calculate_confusion_matrix(preds,target):
    """
    :param preds: prediction array
    :param label: target array
    :return: confusion matrix
    """
    assert preds.shape==target.shape
    return confusion_matrix(target,preds)

def calculate_accuracy(preds,target):
    """
    :param preds: prediction array
    :param label: target array
    :return: accuracy
    """
    assert preds.shape==target.shape
    return (preds==target).sum()

def calculate_f1_measure(preds,target,labels=None):
    """
    :param preds: prediction array
    :param label: target array
    :return: accuracy
    """
    assert preds.shape == target.shape
    return f1_score(y_true=target,y_pred=preds,labels=labels)

def calculate_recall(preds,target,labels=None):
    """
    :param preds: prediction array
    :param label: target array
    :return: recall
    """
    assert preds.shape == target.shape
    return recall_score(y_true=target,y_pred=preds,labels=labels)

def calculate_precision(preds,target,labels=None):
    """
    :param preds: prediction array
    :param label: target array
    :return: precision
    """
    assert preds.shape == target.shape
    return precision_score(y_true=target,y_pred=preds,labels=labels)
