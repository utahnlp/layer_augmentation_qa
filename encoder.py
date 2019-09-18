import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from highway import *
from join_table import *
from dropout_lstm import *
from locked_dropout import *
from char_cnn import *
from char_rnn import *

# encoder
class Encoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Encoder, self).__init__()
		self.opt = opt
		self.shared = shared

		# sampling to hidden_size embeddings
		if opt.word_vec_size != opt.hidden_size:
			self.sampler = nn.Linear(opt.word_vec_size, opt.hidden_size)

		# highway
		self.highway = Highway(opt, opt.hidden_size*2)

		# viewer
		self.context_view = View(1,1)
		self.context_unview = View(1,1,1)
		self.query_view = View(1,1)
		self.query_unview = View(1,1,1)
		self.char_context_view = View(1,1)
		self.char_query_view = View(1,1)

		# dropout for rnn
		self.drop = nn.Dropout(opt.dropout)

		# char encoder
		if opt.char_encoder == 'cnn':
			self.char_encoder = CharCNN(opt, shared)
		elif opt.char_encoder == 'rnn':
			self.char_encoder = CharRNN(opt, shared)
		else:
			assert(False)

		# rnn after highway
		self.bidir = opt.birnn == 1
		rnn_in_size = opt.hidden_size*2	# input size is the output size of highway
		rnn_hidden_size = opt.hidden_size*2 if not self.bidir else opt.hidden_size

		if opt.rnn_type == 'lstm':
			self.rnn = nn.LSTM(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=opt.enc_rnn_layer,
				bias=True,
				batch_first=True,
				dropout=opt.dropout,
				bidirectional=self.bidir)

		elif opt.rnn_type == 'gru':
			self.rnn = nn.GRU(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=opt.enc_rnn_layer,
				bias=True,
				batch_first=True,
				dropout=opt.dropout,
				bidirectional=self.bidir)

		else:
			assert(False)

		self.rnn_joiner = JoinTable(1)
		self.emb_joiner = JoinTable(1)


	def rnn_over(self, seq):
		if self.opt.rnn_type == 'lstm' or self.opt.rnn_type == 'gru':
			E, _ = self.rnn(self.drop(seq))
			return E

		else:
			assert(False)


	def masked_fill_query(self, U):
		max_query_l = self.shared.query_l.max()
		assert(U.shape == (self.shared.batch_l, max_query_l, self.opt.hidden_size*2))

		# build mask that flag paddings with 1
		mask = torch.ones(self.shared.batch_l, max_query_l, 1)
		for i, l in enumerate(self.shared.query_l):
			if l < max_query_l:
				mask[i, l:, :] = 0.0
		mask = Variable(mask, requires_grad=False)
		if self.opt.gpuid != -1:
			mask = mask.cuda()

		return U * mask


	# context of shape (batch_l, context_l, word_vec_size)
	# query of shape (batch_l, query_l, word_vec_size)
	# char_context of shape (batch_l, context_l, token_l, char_emb_size)
	# char_query of shape (batch_l, query_l, token_l, char_emb_size)
	#
	# padding of queries will be wiped out with 0
	def forward(self, context, query, char_context, char_query):
		self.update_context()
		max_query_l = self.shared.query_l.max()

		H = self.context_view(context)
		U = self.query_view(query)

		# sampling word embeddings, optional
		if hasattr(self, 'sampler'):
			H = self.sampler(H)		# (batch_l * context_l, hidden_size)
			U = self.sampler(U)		# (batch_l * query_l, hidden_size)

		# get char encodings
		char_H = self.char_encoder(self.char_context_view(char_context))	# (batch_l * context_l, hidden_size)
		char_U = self.char_encoder(self.char_query_view(char_query))		# (batch_l * query_l, hidden_size)

		H = self.emb_joiner([H, char_H])
		U = self.emb_joiner([U, char_U])

		# highway
		# context will be (batch_l, context_l, hidden_size)
		# query will be (batch_l, query_l, hidden_size)
		context = self.context_unview(self.highway(H))
		query = self.query_unview(self.highway(U))

		# rnn
		H = self.rnn_over(context)
		U = self.rnn_over(query)
		U = self.masked_fill_query(U)	# clear up query paddings

		self.shared.H = H
		self.shared.U = U

		# sanity check
		assert(H.shape == (self.shared.batch_l, self.shared.context_l, self.opt.hidden_size*2))
		assert(U.shape == (self.shared.batch_l, max_query_l, self.opt.hidden_size*2))

		return [self.shared.H, self.shared.U]


	def update_context(self):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		max_query_l = self.shared.query_l.max()
		hidden_size = self.opt.hidden_size

		self.char_context_view.dims = (batch_l * context_l, self.opt.token_l, self.opt.char_emb_size)
		self.context_view.dims = (batch_l * context_l, self.opt.word_vec_size)
		self.context_unview.dims = (batch_l, context_l, hidden_size*2)

		self.char_query_view.dims = (batch_l * max_query_l, self.opt.token_l, self.opt.char_emb_size)
		self.query_view.dims = (batch_l * max_query_l, self.opt.word_vec_size)
		self.query_unview.dims = (batch_l, max_query_l, hidden_size*2)


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





	
