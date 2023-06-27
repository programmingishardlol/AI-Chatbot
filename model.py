import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    # from sentence bag of words convert to number of patterns then return probability of each classes
    # hidden_size is changeable
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)

        # activation function for in-between
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)

        out = self.layer2(out)
        out = self.relu(out)

        out = self.layer3(out)
        return out