import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
    super().__init__()

    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.rnn = nn.RNN(embedding_dim, hidden_dim)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):

    embedded = self.embedding(x)

    output, hidden = self.rnn(embedded)
    assert torch.equal(output[-1,:,:], hidden.squeeze(0))

    out = self.fc(hidden)
    return out

class LSTM(nn.Module):
  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
    super().__init__()

    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.rnn = nn.LSTM(embedding_dim, hidden_dim)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):

    embedded = self.embedding(x)

    output, hidden = self.rnn(embedded)

    out = self.fc(output[-1, :, : ].unsqueeze(0))
    return out

