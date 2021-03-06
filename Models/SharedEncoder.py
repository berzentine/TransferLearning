import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchwordemb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# for now a BiLSTM Encoder

class BiLSTMEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, vocab_size, embed_size):
        super(BiLSTMEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.biLSTM = nn.LSTM(input_size=self.embed_size, hidden_size=hidden_size//2, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, input, hidden):
        emdout = self.embed(input)
        out, hidden = self.biLSTM(emdout, hidden)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (autograd.Variable(weight.new(2*self.num_layers, batch_size, self.hidden_dim//2).zero_()),
                autograd.Variable(weight.new(2*self.num_layers, batch_size, self.hidden_dim//2).zero_()))
