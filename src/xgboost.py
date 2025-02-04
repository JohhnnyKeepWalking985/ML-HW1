from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle, json

class GradientBoostingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, 
                 subsample=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.subsample = subsample
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            subsample=subsample
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "subsample": self.subsample}
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.model = GradientBoostingClassifier(
            n_estimators = self.n_estimators,
            learning_rate = self.learning_rate,
            max_depth = self.max_depth,
            random_state = self.random_state,
            subsample = self.subsample
        )
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, print_result=True):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        if print_result:
            print("XGBOOST Model Accuracy:", acc)
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
        print(f"XGBOOST Model saved to {file_path}")

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"XGBOOST Model loaded from {file_path}")
