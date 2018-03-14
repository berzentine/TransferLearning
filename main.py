from EventClassification.EventClassifier import CNN_Text
from EventClassification.RelevanceClassifier import LogisticRegression
from Models.PrivateEncoderSource import BiLSTMEncoder
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
vocab_size = 1000 # TODO # handle data and reasign vocab size


def plot(train_losses, val_losses, save):
    """ Plots the loss curves """
    df = pd.DataFrame({"Train":train_losses, "Val":val_losses},columns=["Train","Val"])
    df["Epochs"] = df.index
    var_name = "Loss Type"
    value_name = "Loss"
    df = pd.melt(df, id_vars=["Epochs"], value_vars=["Train", "Val"], var_name=var_name, value_name=value_name)
    sns.tsplot(df, time="Epochs", unit=var_name, condition=var_name, value=value_name)
    matplotlib.pyplot.savefig(save, bbox_inches="tight")

##################################################
# Batchify and Handle Data
##################################################
def batchify():
    return 0

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

##################################################
# Train and Evaluate the model
##################################################


def train(train_data):
    model.train()
    hidden = model.init_hidden(num_layers, data_batch_size, hidden_size)

    total_loss = 0
    total_avg_loss = 0
    total_words = 0
    batch_loss = 0
    batch_words = 0
    batch_count = 0
    start_time = time.time()
    for i, data in enumerate(train_data, 0):


    ############################
    # (1) Update Lsim
    ############################
    ############################
    # (2) Update Ldiff
    ############################
    ############################
    # (3) Update Lrec
    ############################
    ############################
    # (4) Update Lrel
    ############################
    ############################
    # (5) Update Lclass
    ############################


    return 0

def evaluate(data, data_length):
    return 0


model_pt = (hidden_size, num_layers, vec, vocab_size, embed_size) # private target
# intializie the weights as netG.apply(weights_init) for all models
model_ps = (hidden_size, num_layers, vec, vocab_size, embed_size) # private source
model_s = (hidden_size, num_layers, vec, vocab_size, embed_size) # encoder shared
model_d = (hidden_size, num_layers, vec, vocab_size, embed_size) # decoder
model_rc = (hidden_size, num_layers, vec, vocab_size, embed_size) # rel classifier
model_fc = (hidden_size, num_layers, vec, vocab_size, embed_size) # final classifier
## setup criterion like this = nn.BCELoss()
# # setup optimizer

# TODO: if cude then do CUDA for
"""
if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    LIKE THIS

    model = BiLSTMEncoder(vocab_size=vocab_size, embed_size=emb_size, num_layers=num_layers, hidden_size=hidden_size)
    if args.cuda:
        model.cuda()
"""

best_val_loss = None
val_losses = []
try:
    for epoch in range(1, total_epochs+1):
        epoch_start_time = time.time()
        train(train_data)
        val_loss = evaluate(val_data, val_lengths)
        val_losses.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(epoch, (time.time() - epoch_start_time),val_loss))
        print('-' * 89)
        #exit(0)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(model_save_path, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 2
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
plot(train_losses, val_losses, plot_save_path)
# Load the best saved model.
with open(model_save_path, 'rb') as f:
    model = torch.load(f)
# Run on test data.
test_loss = evaluate(test_data, test_lengths)
print('=' * 89)
print('| End of training | test loss {:5.2f} | '.format(test_loss))
print('=' * 89)


# Train
"""
m = LogisticRegression(input_size=emb_size, num_classes=2)
print m.forward(Variable(a))
m = CNN_Text(emb_size, vocab_size, dropout) # image_# x n_channel x width x height
a = torch.FloatTensor(32,10,emb_size)   # N X W X D
print m.forward(Variable(a))
"""


print 'done'
