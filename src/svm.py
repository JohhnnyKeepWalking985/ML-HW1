from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import json

class SVMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def get_params(self, deep=True):
        return {"kernel": self.kernel,
                "C": self.C,
                "gamma": self.gamma}
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, print_result=True):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        if print_result:
            print("SVM Model Accuracy:", acc)
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
        print(f"SVM model saved to {file_path}")

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"SVM model loaded from {file_path}")