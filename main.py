from FeatureExtractor import BiLSTMEncoder
from LabelClassifier import CNN_Text
from DomainClassifier import LogisticRegression
import torchwordemb
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# call feature extractor to encode the information
# call classifier to decode and classify based on this information

parser = argparse.ArgumentParser(description='PyTorch Text Generation Model')
parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--batchsize', type=int, default=32,help='batchsize')
parser.add_argument('--lr', type=int, default=0.1,help='learning rate')
parser.add_argument('--data', type=str, default='./data/Wiki-Data/wikipedia-biography-dataset/',help='location of the data corpus')
parser.add_argument('--model_save_path', type=str, default='./saved_models/best_model.pth',help='location of the best model to save')
parser.add_argument('--plot_save_path', type=str, default='./saved_models/loss_plot.png',help='location of the loss plot to save')
parser.add_argument('--emsize', type=int, default=200,help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1,help='number of layers')
parser.add_argument('--nhid', type=int, default=512,help='number of hidden units per layer')
parser.add_argument('--dropout', type=float, default=0.2,help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--clip', type=float, default=0.2,help='gradient clip')
parser.add_argument('--log_interval', type=float, default=500,help='log interval')
parser.add_argument('--epochs', type=int, default=5,help='epochs')

args = parser.parse_args()
cuda = args.cuda
total_epochs = args.epochs
dropout = args.dropout
seed = args.seed
num_layers = args.nlayers
emb_size = args.emsize
hidden_size = args.nhid
batchsize = args.batchsize
data_path = args.data
plot_save_path = args.plot_save_path
model_save_path = args.model_save_path
lr = args.lr
clip = args.clip
log_interval = args.log_interval
vocab_size = 1000
# TODO


# handle data and reasign vocab size

model = BiLSTMEncoder(vocab_size=vocab_size, embed_size=emb_size, num_layers=num_layers, hidden_size=hidden_size)
if args.cuda:
    model.cuda()

# Train
"""
m = LogisticRegression(input_size=emb_size, num_classes=2)
a = torch.FloatTensor(emb_size)
print m.forward(Variable(a))
m = CNN_Text(emb_size, vocab_size, dropout) # image_# x n_channel x width x height
a = torch.FloatTensor(32,10,emb_size)   # N X W X D
print m.forward(Variable(a))
"""

# train and evaluate the model

# dataloader make, for now, just let it be random data




print 'done'
