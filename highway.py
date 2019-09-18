import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *

class HighwayLayer(torch.nn.Module):
	def __init__(self, opt, hidden_size):
		super(HighwayLayer, self).__init__()
		self.opt = opt

		self.drop = nn.Dropout(opt.dropout)
		self.tran_linear = nn.Linear(hidden_size, hidden_size)
		self.gate_linear = nn.Linear(hidden_size, hidden_size)
		self.tran_act = nn.ReLU()
		self.gate_act = nn.Sigmoid()

	# x is of shape (batch_l * seq_l, opt.hidden_size)
	def forward(self, x):
		self.one = Variable(torch.ones(1), requires_grad=False)
		if self.opt.gpuid != -1:
			self.one = self.one.cuda()

		x = self.drop(x)
		tran = self.tran_act(self.tran_linear(x))
		gate = self.gate_act(self.gate_linear(x))

		return gate * tran + (self.one - gate) * x


# Highway networks
class Highway(torch.nn.Module):
	def __init__(self, opt, hidden_size):
		super(Highway, self).__init__()
		self.opt = opt

		hw_layer = opt.hw_layer
		self.hw_layers = nn.ModuleList([HighwayLayer(opt, hidden_size) for _ in range(hw_layer)])

	# input is encoding tensor of shape (batch_l * seq_l, hidden_size)
	def forward(self, seq):
		self.update_context()

		for i, hl in enumerate(self.hw_layers):
			seq = hl(seq)

		return seq


	def update_context(self):
		pass

	def begin_pass(self):
		pass

	def end_pass(self):
		pass

