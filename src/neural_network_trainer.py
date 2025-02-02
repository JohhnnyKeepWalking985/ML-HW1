import itertools
import torch
import pandas as pd
import torch.nn.functional as F
import yaml
import json
from sklearn.metrics import accuracy_score, classification_report
from neural_network import NeuralNetworkModel
from pprint import pprint
import matplotlib.pyplot as plt

class NeuralNetworkTrainer:
    def __init__(self, config_path, device="cpu"):
        self.device = torch.device(device)
        self.config = self.load_config(config_path)

        self.train_data_path = self.config["data"]["train_path"]
        self.test_data_path = self.config["data"]["test_path"]
        self.target_column = self.config["data"]["target_column"]

        self.hidden_sizes = self.config["model"]["hidden_sizes"]
        self.num_layers = self.config["model"]["num_layers"]
        self.activation = self.config["model"]["activation"]
        self.learning_rate = self.config["model"]["learning_rate"]
        self.epochs = self.config["model"]["epochs"]
        self.batch_size = self.config["model"]["batch_size"]
        self.model_save_path = self.config["model"]["save_path"]
        self.eval_save_path = self.config["evaluation"]["save_path"]

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_data(self, data_path):
        df = pd.read_csv(data_path)
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])
        return X, y

    def train_model(self, model, X_train, y_train, epochs, batch_size, print_loss = False):
        model.to(self.device)
        model.train()

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        epoch_loss = 0
        num_batches = 0
        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                model.optimizer.zero_grad()
                outputs = model(X_batch)
                loss = model.criterion(outputs, y_batch)
                loss.backward()
                model.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            if print_loss: 
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_epoch_loss:.4f}")

        torch.save(model.state_dict(), self.model_save_path)

    def evaluate(self, model, X_test, y_test):
        model.to(self.device)
        model.eval()

        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions = torch.argmax(outputs, axis=1).cpu().numpy()

        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        evaluation_results = {
            "test_accuracy": accuracy,
            "classification_report": report
        }
        return evaluation_results
    
    def save_evaluation_results(self, evaluation_results, save_path):
        with open(save_path, 'w') as file:
            json.dump(evaluation_results, file, indent=4)
        print(f"Evaluation results saved to {save_path}")
    
    def train_and_evaluate(self, debug=False):
        print("Loading training data...")
        X_train, y_train = self.load_data(self.train_data_path)
        print("Loading test data...")
        X_test, y_test = self.load_data(self.test_data_path)

        print("Initializing model...")
        model = NeuralNetworkModel(
            input_size=X_train.shape[1],
            hidden_sizes=self.hidden_sizes,
            num_layers=self.num_layers,
            output_size=len(y_train.unique()),
            activation=self.activation,
            learning_rate=self.learning_rate
        )

        if not debug:
            print("Training model...")
            self.train_model(model, X_train, y_train, self.epochs, self.batch_size, print_loss=True)

            print("Evaluating model...")
            evaluation_results = self.evaluate(model, X_test, y_test)
            pprint(evaluation_results['classification_report'], width=50)
            self.save_evaluation_results(evaluation_results, self.eval_save_path)

        if debug:
            training_sizes = []
            train_scores = []
            val_scores = []

            num_splits = 10
            train_size_increment = len(X_train) // num_splits

            for i in range(1, num_splits + 1):
                train_size = train_size_increment * i
                X_train_subset, y_train_subset = X_train[:train_size], y_train[:train_size]

                self.train_model(model, X_train_subset, y_train_subset, self.epochs, self.batch_size, print_loss=False)
                train_score = self.evaluate(model, X_train_subset, y_train_subset)['test_accuracy']
                val_score = self.evaluate(model, X_test, y_test)['test_accuracy']

                training_sizes.append(train_size)
                train_scores.append(train_score)
                val_scores.append(val_score)
            
            self.plot_learning_curve(training_sizes, train_scores, val_scores, metric="accuracy")

    def plot_learning_curve(self, training_sizes, train_scores, val_scores, metric="accuracy"):
        plt.figure(figsize=(8, 6))
        plt.plot(training_sizes, train_scores, marker='o', label=f"Training {metric}")
        plt.plot(training_sizes, val_scores, marker='s', label=f"Validation {metric}")
        
        plt.xlabel("Training Set Size")
        plt.ylabel(metric.capitalize())
        plt.title(f"Learning Curve ({metric.capitalize()})")
        plt.legend()
        plt.grid(True)
        plt.show()

    def tune_hyperparameters(self, param_grid):
        
        X_train, y_train = self.load_data(self.train_data_path)
        X_test, y_test = self.load_data(self.test_data_path)

        param_combinations = list(itertools.product(
            param_grid["num_layers"],
            param_grid["hidden_sizes"],
            param_grid["learning_rate"],
            param_grid["activation"],
            param_grid["epochs"],
            param_grid["batch_size"]
        ))

        best_score = 0
        best_params = None
        best_model = None

        for params in param_combinations:
            num_layers, hidden_sizes, learning_rate, activation, epochs, batch_size = params
            print(f"Testing params: {params}")

            model = NeuralNetworkModel(
                input_size=X_train.shape[1],
                hidden_sizes=hidden_sizes,
                num_layers=num_layers,
                output_size=len(y_train.unique()),
                activation=activation,
                learning_rate=learning_rate
            )

            self.train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
            accuracy = self.evaluate(model, X_test, y_test).get("test_accuracy")
            print(f"Accuracy: {accuracy:.4f}")

            if accuracy > best_score:
                best_score = accuracy
                best_params = params
                best_model = model

        print(f"\nBest Hyperparameters: {best_params}")
        print(f"Best Accuracy: {best_score:.4f}")
        return best_model, best_params