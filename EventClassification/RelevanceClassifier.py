# this will be a logistic regression
# classifier C2
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    # num_classes = 0/1 0 if no keywords are present else 1    

    def forward(self, x):
        out = self.linear(x)
        return out
