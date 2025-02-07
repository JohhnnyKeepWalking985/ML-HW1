import yaml
import os
import pandas as pd
import numpy as np
from knn import KNNModel 
from svm import SVMModel
from xgboost import GradientBoostingModel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from pathlib import Path
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def resolve_path(base_path, relative_path):
    return os.path.abspath(os.path.join(base_path, relative_path))

def load_data(data_path, target_column):
    df = pd.read_csv(data_path)
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y

def train_model(config_path, debug=False):
    config = load_config(config_path)

    project_dir = Path(config_path).parent.parent.resolve()
    train_data_path = resolve_path(project_dir, config['data']['train_path'])
    test_data_path = resolve_path(project_dir, config['data']['test_path'])
    model_save_path = resolve_path(project_dir, config['model']['save_path'])
    eval_save_path = resolve_path(project_dir, config['evaluation']['save_path'])
    metric = "accuracy"

    target_column = config['data']['target_column']

    print(f"Loading training data from {train_data_path}...")
    X_train, y_train = load_data(train_data_path, target_column)
    print(f"Loading test data from {test_data_path}...")
    X_test, y_test = load_data(test_data_path, target_column)

    model_name = config['model']['name']
    print(f"Initializing the {model_name} model...")
    if model_name == 'knn':
        model = KNNModel(n_neighbors=config['model']['params']['n_neighbors'])
    elif model_name == 'svm':
        model = SVMModel(
            kernel=config['model']['params']['kernel'], 
            C=config['model']['params']['C'], 
            gamma=config['model']['params']['gamma']
        )
    elif model_name == 'xgboost':
        model = GradientBoostingModel(
            n_estimators=config['model']['params']['n_estimators'], 
            learning_rate=config['model']['params']['learning_rate'], 
            max_depth=config['model']['params']['max_depth'],
            subsample=config['model']['params']['subsample'],
            min_samples_leaf=config['model']['params']['min_samples_leaf'],
            n_iter_no_change=config['model']['params']['n_iter_no_change']
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not debug:
        print("Training the model...")
        model.train(X_train, y_train)

        print("Evaluating the model...")
        evaluation_results = model.evaluate(X_test, y_test)
        model.save_evaluation_results(evaluation_results, eval_save_path)
        model.save_model(model_save_path)

    if debug:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, scoring=metric, train_sizes=np.linspace(0.1, 1.0, 20)
        )
        
        plot_learning_curve(train_sizes, train_scores, val_scores, metric=metric)

def plot_learning_curve(train_sizes, train_scores, val_scores, metric="Accuracy"):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Curve")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation Curve")
    plt.xlabel("Training Size")
    plt.ylabel(metric)
    plt.title("Learning Curve with Variance")
    plt.legend()
    plt.show()

def hyper_parameter_tuning(config_path, param_grids, search_method="grid", metric='accuracy', validation_curve=False, param_name=None):
    config = load_config(config_path)

    project_dir = Path(config_path).parent.parent.resolve()
    train_data_path = resolve_path(project_dir, config['data']['train_path'])
    test_data_path = resolve_path(project_dir, config['data']['test_path'])
    eval_save_path = resolve_path(project_dir, config['evaluation']['hyperparameter_tuning_save_path'])

    target_column = config['data']['target_column']

    print(f"Loading training data from {train_data_path}...")
    X_train, y_train = load_data(train_data_path, target_column)
    print(f"Loading test data from {test_data_path}...")
    X_test, y_test = load_data(test_data_path, target_column)

    model_name = config['model']['name']
    print(f"Initializing the {model_name} model for hyperparameter tuning...")

    model_classes = {
        "knn": KNNModel,
        "svm": SVMModel,
        "xgboost": GradientBoostingModel
    }
    model_class = model_classes[model_name]

    if model_name == 'knn':
        model = KNNModel()
    elif model_name == 'svm':
        model = SVMModel()
    elif model_name == 'xgboost':
        model = GradientBoostingModel()
    
    param_grid = param_grids[model_name]

    if search_method == "grid":
        search = GridSearchCV(model, param_grid, cv=3, scoring=metric, n_jobs=-1, verbose=1)
    elif search_method == "random":
        search = RandomizedSearchCV(model, param_grid, cv=3, scoring=metric, n_jobs=-1, verbose=1, n_iter=5)
    else:
        raise ValueError("search_method must be 'grid' or 'random'")

    print("Performing hyperparameter tuning...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    print(f"Best hyperparameters: {best_params}")
    print(f"Best cross-validation {metric}: {best_score}")

    print("Retraining the model with the best hyperparameters...")
    best_model = model_class(**best_params)
    best_model.train(X_train, y_train)

    print("Evaluating the best model on the test set...")
    evaluation_results = best_model.evaluate(X_test, y_test)

    hyper_parameter_tuning_results = {
        "best_params": best_params,
        "best_score": best_score
    }
    hyper_parameter_tuning_results.update(evaluation_results)
    model.save_evaluation_results(hyper_parameter_tuning_results, eval_save_path)
    print(f"Results saved to {eval_save_path}")

    if validation_curve:
        param_values = param_grid[param_name]
        plot_validation_curve(model_class, X_train, y_train, X_test, y_test, param_name, param_values, metric)

    return best_model, best_params, best_score

def plot_validation_curve(model_class, X_train, y_train, X_test, y_test, param_name, param_values, metric):
    train_scores = []
    val_scores = []

    for value in param_values:
        model_params = {param_name: value}
        model = model_class(**model_params)

        model.train(X_train, y_train)
        train_acc = model.evaluate(X_train, y_train, print_result=False)["test_accuracy"]
        val_acc = model.evaluate(X_test, y_test, print_result=False)["test_accuracy"]
        train_scores.append(train_acc)
        val_scores.append(val_acc)

    plt.figure(figsize=(8, 6))
    plt.plot(param_values, train_scores, marker='o', linestyle='-', label="Training metric")
    plt.plot(param_values, val_scores, marker='s', linestyle='--', label="Validation metric")
    plt.xlabel(param_name)
    plt.ylabel(metric)
    plt.title(f"Validation Curve for {param_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    config_path = Path(__file__).parent / "configs" / "knn.yaml"
    train_model(config_path)
