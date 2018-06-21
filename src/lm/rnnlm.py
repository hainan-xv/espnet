from __future__ import division, print_function
import time
import sys
import random
import argparse
#import pdb

from itertools import count
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--MB_SIZE', type=int, help='minibatch size')
parser.add_argument('--EMBED_SIZE', type=int, help='embedding size')
parser.add_argument('--HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('--EPOCHS', type=int, help='number of epochs')

parser.add_argument('--TRAIN', type=str, help='train text')
parser.add_argument('--VALID', type=str, help='valid text')
parser.add_argument('--VOCAB', type=str, help='vocab text')

parser.add_argument('--CUDA', action='store_true', help='use CUDA')
parser.add_argument('--gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA")
args = parser.parse_args()

torch.cuda.set_device(args.gpus[0])

train_file = args.TRAIN
valid_file = args.VALID
vocab_file = args.VOCAB

def read(fname, unk_index):
  """
  Read a file where each line is of the form "word1 word2 ..."
  Yields lists of the form [word1, word2, ...]
  """
  with open(fname, "r") as fh:
    for line in fh:
      sent = [w2i.get(x, unk_index) for x in line.strip().split()]
      sent.append(w2i["<s>"])
      yield torch.LongTensor(sent)

def read_vocab(fname):
  w2i = {}
  w2i["<MASK>"] = 0
  with open(fname, "r") as f:
    i = 2
    for line in f:
      w2i[line.strip()] = i
      i = i + 1
  w2i["<RNN_UNK>"] = i
  unk_id = i
  return w2i, unk_id

w2i, unk_index = read_vocab(vocab_file)
mask = w2i['<MASK>']
assert mask == 0
train = list(read(train_file, unk_index))
valid = list(read(valid_file, unk_index))
#test  = list(read(test_file))
vocab_size = len(w2i)
S = w2i['<s>']
print ("vocab size is ", vocab_size)

def get_batch(sequences, volatile=False):
  lengths = torch.LongTensor([len(s) for s in sequences])
  batch   = torch.LongTensor(lengths.max(), len(sequences)).fill_(mask)
  for i, s in enumerate(sequences):
    batch[:len(s), i] = s
  if args.CUDA:
    batch = batch.cuda()
  return Variable(batch, volatile=volatile), lengths

# class RNNLM(nn.Module):
#   def __init__(self):
#     super(RNNLM, self).__init__()
#     self.embeddings = nn.Embedding(vocab_size, args.EMBED_SIZE)
#     self.rnn = nn.LSTM(args.EMBED_SIZE, args.HIDDEN_SIZE)
#     self.proj = nn.Linear(args.HIDDEN_SIZE, vocab_size)
#     self.proj.weight.data = self.embeddings.weight.data  # typing
#     self.embeddings.weight.requires_grad=False
#     self.embeddings.weight.data = torch.randn(vocab_size, args.EMBED_SIZE) * 10
#   def forward(self, sequences):
#     rnn_output, _ = self.rnn(self.embeddings(sequences))
#     return self.proj(rnn_output.view(-1, args.HIDDEN_SIZE))
class RNNLM(nn.Module):
    def __init__(self, n_vocab, n_units, n_layers, dropout):
        super(RNNLM, self).__init__()
        self.n_vocab = n_vocab
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.embed = torch.nn.Embedding(n_vocab, n_units)
        self.rnn = nn.LSTM(n_units, n_units, num_layers=n_layers, dropout=dropout)
        self.d = torch.nn.Dropout(p=dropout)
#        self.l1 = torch.nn.LSTMCell(n_units, n_units)
#        self.d1 = torch.nn.Dropout()
#        self.l2 = torch.nn.LSTMCell(n_units, n_units)
#        self.d2 = torch.nn.Dropout()
        self.lo = torch.nn.Linear(n_units, n_vocab)

        # initialize parameters from uniform distribution
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

        # tying
        self.embed.weight.data = self.lm.weight.data

    def zero_state(self, batchsize):
        return Variable(torch.zeros(self.n_layers, batchsize, self.n_units)).float()

    def forward(self, state, x):
        if state is None:
            state = (to_cuda(self, self.zero_state(x.size(0))),
                     to_cuda(self, self.zero_state(x.size(0))))
        h0 = self.p(self.embed(x))
#        h1, c1 = self.l1(self.d0(h0), (state['h1'], state['c1']))
#        h2, c2 = self.l2(self.d1(h1), (state['h2'], state['c2']))
        output, out_state = self.rnn(h0, state)
        output = p(output)
        y = self.lo(output)
#        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
        return out_state, y
     def seq_forward(self, sequences):
         rnn_output, _ = self.rnn(self.embeddings(sequences))
         return self.proj(rnn_output.view(-1, args.HIDDEN_SIZE))


# build the model
rnnlm = RNNLM()
#parameters = list(filter(lambda p: p.requires_grad, rnnlm.parameters()))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, rnnlm.parameters()), lr=0.01)
weight = torch.FloatTensor(vocab_size).fill_(1)
weight[mask] = 0
loss_fn = nn.CrossEntropyLoss(weight, size_average=False)

if args.CUDA:
  rnnlm.cuda()
  loss_fn.cuda()

# Sort training sentences in descending order and count minibatches
# train.sort(key=lambda x: -len(x))
# valid.sort(key=lambda x: -len(x))
#test.sort(key=lambda x: -len(x))

train_order = range(0, len(train), args.MB_SIZE)  # [x*args.MB_SIZE for x in range(int((len(train)-1)/args.MB_SIZE + 1))]
valid_order = range(0, len(valid), args.MB_SIZE)  # [x*args.MB_SIZE for x in range(int((len(train)-1)/args.MB_SIZE + 1))]
#test_order  = range(0, len(test), args.MB_SIZE)  # [x*args.MB_SIZE for x in range(int((len(test)-1)/args.MB_SIZE + 1))]

# Perform training
print("startup time: %r" % (time.time() - start))
start = time.time()
i = total_time = dev_time = total_tagged = current_words = current_loss = 0

file_id = 0
for ITER in range(args.EPOCHS):
  i = 0
  print ("starting epoch", ITER)
  random.shuffle(list(train_order))
  for sid in train_order:
    i += 1
    # train
    batch, lengths = get_batch(train[sid:sid + args.MB_SIZE])
    scores = rnnlm(batch[:-1])
    loss = loss_fn(scores, batch[1:].view(-1))
    # optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # log loss
    current_words += lengths.sum() - lengths.size(0)  # ignore <s>
    current_loss += loss.data[0]

    if i % int(40000 / args.MB_SIZE) == 0:
      print(" training %.1f%% train PPL=%.4f" % (i / len(train_order) * 100, np.exp(current_loss / current_words)))
      total_tagged += current_words
      current_loss = current_words = 0
      total_time = time.time() - start
    # log perplexity
    if i % int(40000 / args.MB_SIZE) == 0:
      dev_start = time.time()
      dev_loss = dev_words = 0
      for j in valid_order:
        batch, lengths = get_batch(valid[j:j + args.MB_SIZE], volatile=True)
        scores = rnnlm(batch[:-1])
        dev_loss += loss_fn(scores, batch[1:].view(-1)).data[0]
        dev_words += lengths.sum() - lengths.size(0)  # ignore <s>
      dev_time += time.time() - dev_start
      train_time = time.time() - start - dev_time
      print("           dev   PPL=%.4f word_per_sec=%.4f" % (
          np.exp(dev_loss / dev_words), total_tagged / train_time))
#      print("  nll=%.4f, ppl=%.4f, words=%r, time=%.4f, word_per_sec=%.4f" % (
#          dev_loss / dev_words, np.exp(dev_loss / dev_words), dev_words, train_time, total_tagged / train_time))
      torch.save(rnnlm.state_dict(), "%r.mdl" % file_id)
      file_id = file_id + 1

  print("epoch %r finished" % ITER)
