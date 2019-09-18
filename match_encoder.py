import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from join_table import *
from locked_dropout import *
from self_attention import *

# match encoder used in BIDAF+Elmo
class MatchEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(MatchEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		# dropout for rnn
		self.drop = LockedDropout(opt.dropout)

		enc_size = opt.hidden_size*2 if opt.use_elmo_post == 0 else opt.hidden_size*2 + opt.elmo_size
		self.linear1_in_size = enc_size * 4
		self.linear1 = nn.Sequential(
			nn.Linear(self.linear1_in_size, opt.hidden_size*2),
			nn.ReLU())
		self.linear1_view = View(1,1,1)
		self.linear1_unview = View(1,1)

		self.linear2_in_size = opt.hidden_size * 6
		self.linear2 = nn.Sequential(
			nn.Linear(self.linear2_in_size, opt.hidden_size*2),
			nn.ReLU())
		self.linear2_view = View(1,1,1)
		self.linear2_unview = View(1,1)

		#
		self.phi_joiner = JoinTable(2)

		# match_encoder is only supposed to work with self_att
		assert(opt.self_att == 'self_att')
		self.self_attention = SelfAttention(opt, shared, opt.hidden_size*2, prod_type='trilinear', mask_type='diagonal')

		bidir = opt.birnn == 1
		rnn_in_size = opt.hidden_size*2
		rnn_hidden_size = opt.hidden_size*2 if not bidir else opt.hidden_size
		self.rnn = build_rnn(
			opt.rnn_type,
			input_size=rnn_in_size,
			hidden_size=rnn_hidden_size,
			num_layers=opt.reenc_rnn_layer,
			bias=True,
			batch_first=True,
			dropout=opt.dropout,
			bidirectional=bidir)


	def rnn_over(self, x):
		x = self.drop(x)
		x, _ = self.rnn(x)
		x = self.drop(x)
		return x


	def forward(self, G):
		self.update_context()

		# downsample to a managable size
		G = self.linear1_unview(self.linear1(self.linear1_view(G)))
		# rnn encode
		phi = self.rnn_over(G)

		# selfattention
		attended = self.self_attention(phi)
		phi = self.phi_joiner([phi, attended, phi * attended])

		# bottle end downsampling
		phi = self.linear2_unview(self.linear2(self.linear2_view(phi)))

		# bottle residual
		M = G + phi

		self.shared.M = M

		return M


	def update_context(self):
		self.linear1_view.dims = (self.shared.batch_l * self.shared.context_l, self.linear1_in_size)
		self.linear1_unview.dims = (self.shared.batch_l, self.shared.context_l, self.opt.hidden_size*2)
		self.linear2_view.dims = (self.shared.batch_l * self.shared.context_l, self.linear2_in_size)
		self.linear2_unview.dims = self.linear1_unview.dims

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





	
