import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from math import sqrt
import pdb


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.fc1 = nn.Linear(2 * len(Ks) * Co, 1)

        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x, y):
        #pdb.set_trace()
        x = self.embed(x)  # (N, W, D)

        y = self.embed(y)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        y = y.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        y = [F.relu(conv(y)).squeeze(3) for conv in self.convs]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        y = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]

        x = torch.cat(x, 1)

        y = torch.cat(y, 1)

        comparison_vector = torch.cat([torch.abs(x-y), x*y], 1)

        logit = self.fc1(comparison_vector)

        logit = F.sigmoid(logit)
        return logit

class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, num_heads=2):
        super(SelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # The shape of x is (batch, len, dim_q)
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v // nh

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)

        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # (batch, len, dim_v)
        return att

class LSTM_Text(nn.Module):
    def __init__(self, args):
        super(LSTM_Text, self).__init__()
        self.embedding_dim = args.embed_dim
        self.hidden_dim = args.hid_dim
        self.vocab_size = args.embed_num
        self.use_attention = args.use_att
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=args.dropout)
        self.rnn = nn.LSTM(input_size=self.embedding_dim,
                           hidden_size=self.hidden_dim, num_layers=args.num_layers,
                           bidirectional=False if args.num_dir == 1 else True,
                           batch_first=True)
        lstm_output_dim = args.num_dir * self.hidden_dim
        self.attention = SelfAttention(dim_q=lstm_output_dim,
                                       dim_k=lstm_output_dim, dim_v=lstm_output_dim)
        self.fc1 = nn.Linear(2 * lstm_output_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_output_dim, lstm_output_dim),
            nn.ReLU(),
            nn.Linear(lstm_output_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, *kargs):
        assert len(kargs) == 2, "there must be 2 inputs"
        logits = []
        for arg in kargs:
            lengths = [s.size(0) for s in arg]
            x = self.embed(arg)
            x_packed_input = pack_padded_sequence(input=x, lengths=lengths,
                                                  batch_first=True, enforce_sorted=False)
            x, _ = self.rnn(x_packed_input)
            lstm_out, _ = pad_packed_sequence(x, batch_first=True)
            if self.use_attention:
                lstm_out = self.attention(lstm_out)
            logit = lstm_out[:, -1, :]
            logits.append(logit)
        x, y = logits[0], logits[1]
        comparison_vector = torch.cat([torch.abs(x - y), x * y], 1)
        logit = self.fc(comparison_vector)
        # logit = torch.sigmoid(logit)
        return logit


