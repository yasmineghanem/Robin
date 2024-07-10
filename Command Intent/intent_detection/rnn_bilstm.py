import torch
from torch import nn
import torch.nn.functional as F


class IntentDetection(nn.Module):
    def __init__(self, vocab_size, word_embedding_dim, hidden_dim, output_dim, num_layers=1):

        super(IntentDetection, self).__init__()
        # paramaters

        # layers
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the last output of the sequence
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
