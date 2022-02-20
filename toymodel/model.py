import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence


class MyLSTM(nn.Module):
    def __init__(self, seq_length = 6, hidden_dim = 10, dropout=0.4):
        super(MyLSTM, self).__init__()
        self.seq_length = seq_length 
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.dropout = dropout
        self.lstm = nn.LSTM(self.seq_length, self.hidden_dim, self.num_layers,batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout)
        self.linear1 = nn.Linear(self.hidden_dim, 5)
        self.linear2 = nn.Linear(5,3)
        self.linear3 = nn.Linear(3,1)
        self.linear4 = nn.Linear(3,1)
        self.F = nn.ReLU()
    


    def forward(self, inp):
#       
        inp=inp.unsqueeze(0)
        out, hs = self.lstm(inp)
#        out = self.dropout(out)
        # feed only the last hidden state
        out = self.linear1(out[:,-1,:])
        out = self.F(out)
        out = self.linear2(out)
        out = self.F(out)
        out = self.linear3(out)
        out = self.F(out)
#        out = self.linear4(out)
#        out = self.F(out)
        return out

