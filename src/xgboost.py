from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle, json
from collections import Counter
import numpy as np

class GradientBoostingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, 
                 subsample=1, min_samples_leaf=1, n_iter_no_change=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.n_iter_no_change = n_iter_no_change
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            n_iter_no_change=n_iter_no_change
        )

    def train(self, X_train, y_train):
        class_counts = Counter(y_train)
        total_samples = sum(class_counts.values())
        sample_weights = np.array([total_samples / class_counts[label] for label in y_train])
        self.fit(X_train, y_train, sample_weight=sample_weights)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    
    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "subsample": self.subsample,
                "min_samples_leaf": self.min_samples_leaf,
                "n_iter_no_change": self.n_iter_no_change}
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.model = GradientBoostingClassifier(
            n_estimators = self.n_estimators,
            learning_rate = self.learning_rate,
            max_depth = self.max_depth,
            random_state = self.random_state,
            subsample = self.subsample,
            min_samples_leaf = self.min_samples_leaf,
            n_iter_no_change = self.n_iter_no_change
        )
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, print_result=True):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred)
        if print_result:
            print("XGBOOST Model Accuracy:", acc)
            print(classification_report(y_test, y_pred))
            print("XGBOOST F1 Score:", f1)

        evaluation_results = {
            "test_accuracy": acc,
            "classification_report": report,
            "f1": f1
        }
        return evaluation_results

    def save_evaluation_results(self, evaluation_results, save_path):
        with open(save_path, 'w') as file:
            json.dump(evaluation_results, file, indent=4)
        print(f"Evaluation results saved to {save_path}")


    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"XGBOOST Model saved to {file_path}")

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"XGBOOST Model loaded from {file_path}")
