import torch.nn as nn
import torch.optim as optim

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, output_size, activation='relu', learning_rate=0.001):
        super(NeuralNetworkModel, self).__init__()

        layers = []
        in_features = input_size

        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_sizes))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            in_features = hidden_sizes

        layers.append(nn.Linear(in_features, output_size))

        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)