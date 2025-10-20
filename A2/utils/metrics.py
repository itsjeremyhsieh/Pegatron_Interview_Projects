from sklearn.metrics import accuracy_score

def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy between true labels and predicted answers.
    """
    return accuracy_score(y_true, y_pred)
