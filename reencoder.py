import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from join_table import *
from dropout_lstm import *
from locked_dropout import *

# re-encoder
class ReEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(ReEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		# dropout for rnn
		self.drop = nn.Dropout(opt.dropout)

		self.bidir = opt.birnn == 1
		rnn_in_size = opt.hidden_size*2 * 4
		rnn_hidden_size = opt.hidden_size*2 if not self.bidir else opt.hidden_size
		self.rnn = []
		# do it this way rather than have a 2-layer lstm
		#	pytorch multi-layer lstm is volatile to randomness
		for i in range(opt.reenc_rnn_layer):
			self.rnn.append(
				build_rnn(opt.rnn_type,
					input_size=rnn_in_size if i == 0 else rnn_hidden_size*2,
					hidden_size=rnn_hidden_size, 
					num_layers=1,
					bias=True,
					batch_first=True,
					dropout=opt.dropout,
					bidirectional=self.bidir))
		self.rnn = nn.ModuleList(self.rnn)


	def rnn_over(self, context):
		if self.opt.rnn_type == 'lstm' or self.opt.rnn_type == 'gru':
			M = context
			for i in range(self.opt.reenc_rnn_layer):
				M, _ = self.rnn[i](self.drop(M))
			return M
		else:
			assert(False)


	def forward(self, G):
		self.update_context()

		M = self.rnn_over(G)

		self.shared.M = M

		return M


	def update_context(self):
		pass

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





	
