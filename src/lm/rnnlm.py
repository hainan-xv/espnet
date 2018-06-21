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


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='minibatch size', default=64)
parser.add_argument('--embed_size', type=int, help='embedding size', default=512)
parser.add_argument('--epochs', type=int, help='number of epochs', default=10)
parser.add_argument('--n_layers', type=int, help='number of layers', default=2)
parser.add_argument('--dropout', type=float, help='dropout', default=0.8)

parser.add_argument('--train_label', type=str, help='train text')
parser.add_argument('--valid_label', type=str, help='valid text')
parser.add_argument('--vocab_file', type=str, help='vocab text')

parser.add_argument('--out_dir', type=str, help='output directory')

parser.add_argument('--CUDA', action='store_true', help='use CUDA', default=False)
parser.add_argument('--gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA")
args = parser.parse_args()

def read(fname, unk_index, w2i):
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
  w2i["<eps>"] = 0
  with open(fname, "r") as f:
    i = 1
    for line in f:
      [word, key] = line.split()
      key = int(key)
      w2i[word] = key
      assert(key == i)
      i = i + 1
  w2i["<s>"] = i
  w2i["</s>"] = i + 1
  w2i["<MASK>"] = i + 2
  mask_id = i + 2
  assert(len(w2i) == i + 3)
  return w2i, mask_id

def get_batch(sequences, mask, volatile=False, cuda=False):
  lengths = torch.LongTensor([len(s) for s in sequences])
  batch   = torch.LongTensor(lengths.max(), len(sequences)).fill_(mask)
  for i, s in enumerate(sequences):
    batch[:len(s), i] = s
  if cuda:
    batch = batch.cuda()
  return Variable(batch, volatile=volatile), lengths

class RNNLM(nn.Module):
    def __init__(self, n_vocab, n_units, n_layers, dropout):
        super(RNNLM, self).__init__()
        self.n_vocab = n_vocab
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.embed = torch.nn.Embedding(n_vocab, n_units)
        self.rnn = nn.LSTM(n_units, n_units, num_layers=n_layers, dropout=dropout)
        self.p = torch.nn.Dropout(p=dropout)
        self.lo = torch.nn.Linear(n_units, n_vocab)

        # initialize parameters from uniform distribution
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

        # tying
        self.embed.weight.data = self.lo.weight.data

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

    def seq_score(self, sequences):
        emb = self.p(self.embed(sequences))
        rnn_output, _ = self.rnn(emb)
        rnn_output = self.p(rnn_output)
        return self.lo(rnn_output.view(-1, self.n_units))

def train(args):
  start = time.time()
  if args.CUDA:
    torch.cuda.set_device(args.gpus[0])

  train_file = args.train_label
  valid_file = args.valid_label
  vocab_file = args.vocab_file

  batch_size = args.batch_size
  epochs = args.epochs

  n_units = args.embed_size
  dropout = args.dropout
  n_layers = args.n_layers

  w2i, unk_index = read_vocab(vocab_file)
  mask = w2i['<MASK>']
  train = list(read(train_file, unk_index, w2i))
  valid = list(read(valid_file, unk_index, w2i))
#test  = list(read(test_file))
  vocab_size = len(w2i)
  S = w2i['<s>']
  print ("vocab size is ", vocab_size)


# build the model
  rnnlm = RNNLM(vocab_size, n_units, n_layers, dropout)
#parameters = list(filter(lambda p: p.requires_grad, rnnlm.parameters()))
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, rnnlm.parameters()), lr=0.01)
  weight = torch.FloatTensor(vocab_size).fill_(1)
  weight[mask] = 0
  loss_fn = nn.CrossEntropyLoss(weight, size_average=False)

  cuda = False
  if args.CUDA:
    rnnlm.cuda()
    loss_fn.cuda()
    cuda = True

  train_order = range(0, len(train), batch_size)
  valid_order = range(0, len(valid), batch_size)

# Perform training
  print("startup time: %r" % (time.time() - start))
  start = time.time()
  i = total_time = dev_time = total_tagged = current_words = current_loss = 0

  file_id = 0
  for ITER in range(args.epochs):
    i = 0
    print ("starting epoch", ITER)
    random.shuffle(list(train_order))
    for sid in train_order:
      i += 1
      # train
      batch, lengths = get_batch(train[sid:sid + batch_size], mask, cuda=cuda)
      scores = rnnlm.seq_score(batch[:-1])
      loss = loss_fn(scores, batch[1:].view(-1))
      # optimization
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm(rnnlm.parameters(), 1.5)
      optimizer.step()
      # log loss
      current_words += lengths.sum() - lengths.size(0)  # ignore <s>
      current_loss += loss.data[0]

      if i % int(40000 / batch_size) == 0:
        print(" training %.1f%% train PPL=%.4f" % (i / len(train_order) * 100, np.exp(current_loss / current_words)))
        total_tagged += current_words
        current_loss = current_words = 0
        total_time = time.time() - start
      # log perplexity
      if i % int(200000 / batch_size) == 0:
        dev_start = time.time()
        dev_loss = dev_words = 0
        for j in valid_order:
          batch, lengths = get_batch(valid[j:j + batch_size], mask, volatile=True, cuda=cuda)
          scores = rnnlm.seq_score(batch[:-1])
          dev_loss += loss_fn(scores, batch[1:].view(-1)).data[0]
          dev_words += lengths.sum() - lengths.size(0)  # ignore <s>
        dev_time += time.time() - dev_start
        train_time = time.time() - start - dev_time
        print("           dev   PPL=%.4f word_per_sec=%.4f" % (
            np.exp(dev_loss / dev_words), total_tagged / train_time))
        torch.save(rnnlm.state_dict(), args.out_dir + "/%r.mdl" % file_id)
        file_id = file_id + 1

    print("epoch %r finished" % ITER)

train(args)
