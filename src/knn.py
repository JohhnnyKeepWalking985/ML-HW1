from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json

class KNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors}
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, print_result=True):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        if print_result:
            print("KNN Model Accuracy:", acc)
            print(classification_report(y_test, y_pred))

        evaluation_results = {
            "test_accuracy": acc,
            "classification_report": report
        }
        return evaluation_results
    
    def save_evaluation_results(self, evaluation_results, save_path):
        with open(save_path, 'w') as file:
            json.dump(evaluation_results, file, indent=4)
        print(f"Evaluation results saved to {save_path}")

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"KNN model saved to {file_path}")

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"KNN model loaded from {file_path}")