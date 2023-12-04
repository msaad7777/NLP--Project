import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


def load_data(path):
    data = pd.read_csv(path)
    print(data.head())
    print(data.info())
    return data

def print_results(y_test, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
