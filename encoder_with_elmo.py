import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from highway import *
from join_table import *
from locked_dropout import *
from char_cnn import *
from char_rnn import *
#from elmo_encoder import *
from elmo_loader import *
from var_rnn import *

# encoder with Elmo
class EncoderWithElmo(torch.nn.Module):
	def __init__(self, opt, shared):
		super(EncoderWithElmo, self).__init__()
		self.opt = opt
		self.shared = shared

		self.char_context_view = View(1,1,1)
		self.char_context_unview = View(1,1)
		self.char_query_view = View(1,1,1)
		self.char_query_unview = View(1,1)

		self.elmo_drop = nn.Dropout(opt.elmo_dropout)
		self.drop = LockedDropout(opt.dropout)

		self.phi_joiner = JoinTable(2)

		if opt.use_char_enc == 1:
			if opt.char_encoder == 'cnn':
				self.char_encoder = CharCNN(opt, shared)
			elif opt.char_encoder == 'rnn':
				self.char_encoder = CharRNN(opt, shared)
			else:
				assert(False)

		#if opt.dynamic_elmo == 1:
		#	self.elmo = ElmoEncoder(opt, shared)
		#else:
		#	self.elmo = ElmoLoader(opt, shared)
		self.elmo = ElmoLoader(opt, shared)

		# rnn merger
		bidir = opt.birnn == 1
		rnn_in_size = opt.word_vec_size + opt.char_enc_size + opt.elmo_size
		if opt.use_char_enc == 0:
			rnn_in_size -= opt.char_enc_size
		rnn_hidden_size = opt.hidden_size*2 if not bidir else opt.hidden_size
		self.rnn = VarRNN(build_rnn(
			opt.rnn_type,
			input_size=rnn_in_size,
			hidden_size=rnn_hidden_size,
			num_layers=opt.enc_rnn_layer,
			bias=True,
			batch_first=True,
			dropout=opt.dropout,
			bidirectional=bidir))

		self.gamma_pre = nn.Parameter(torch.ones(1), requires_grad=True)
		self.gamma_pre.skip_init = 1
		self.gamma_post = nn.Parameter(torch.ones(1), requires_grad=True)
		self.gamma_post.skip_init = 1
	
		self.w_pre = nn.Parameter(torch.ones(3), requires_grad=True)
		self.w_pre.skip_init = 1
		self.w_post = nn.Parameter(torch.ones(3), requires_grad=True)
		self.w_post.skip_init = 1

		self.softmax = nn.Softmax(0)


	def masked_fill_query(self, Q):
		return Q * self.shared.query_mask.unsqueeze(-1)


	def rnn_over(self, x, x_len, hidden):
		x = self.drop(x)
		x, h = self.rnn(x, x_len, hidden)
		return x, h


	def concat(self, word, char, elmo):
		if char is not None:
			assert(self.opt.use_char_enc == 1)
			return self.phi_joiner([word, char, elmo])
		assert(self.opt.use_char_enc == 0)
		return self.phi_joiner([word, elmo])


	def sample_elmo(self, sampler, elmo1, elmo2):
		elmo1 = sampler(elmo1.view(-1, self.opt.elmo_in_size*3)).view(self.shared.batch_l, self.shared.sent_l1, -1)
		elmo2 = sampler(elmo2.view(-1, self.opt.elmo_in_size*3)).view(self.shared.batch_l, self.shared.sent_l2, -1)
		return elmo1, elmo2

	def interpolate_elmo(self, elmo_layers1, elmo_layers2, w, gamma):
		# interpolate
		weights = nn.Softmax(0)(w)
		sent1 = elmo_layers1[0] * weights[0] + elmo_layers1[1] * weights[1] + elmo_layers1[2] * weights[2]
		sent2 = elmo_layers2[0] * weights[0] + elmo_layers2[1] * weights[1] + elmo_layers2[2] * weights[2]
		return sent1*gamma, sent2*gamma


	# context of shape (batch_l, context_l, word_vec_size)
	# query of shape (batch_l, max_query_l, word_vec_size)
	# char_context of shape (batch_l, context_l, token_l, char_emb_size)
	# char_query of shape (batch_l, max_query_l, token_l, char_emb_size)
	def forward(self, context, query, char_context, char_query):
		self.update_context()
		max_query_l = self.shared.query_l.max()

		if self.opt.use_char_enc == 1:
			char_context = self.char_context_unview(self.char_encoder(self.char_context_view(char_context)))
			char_query = self.char_query_unview(self.char_encoder(self.char_query_view(char_query)))
		else:
			char_context, char_query = None

		# elmo pass
		elmo1, elmo2 = self.elmo()

		# pre-rnn elmo
		elmo_pre1, elmo_pre2 = self.interpolate_elmo(elmo1, elmo2, self.w_pre, self.gamma_pre)
		elmo_pre1, elmo_pre2 = self.elmo_drop(elmo_pre1), self.elmo_drop(elmo_pre2)

		# concat with word_vec and char_enc
		context = self.concat(context, char_context, elmo_pre1)
		query = self.concat(query, char_query, elmo_pre2)
		query = self.masked_fill_query(query)

		# merge using rnn
		context, _ = self.rnn_over(context, None, None)
		query, _ = self.rnn_over(query, self.shared.query_l, None)
		
		context = self.drop(context)
		query = self.drop(query)

		# get the post-rnn elmo if requires
		if self.opt.use_elmo_post == 1:
			elmo_post1, elmo_post2 = self.interpolate_elmo(elmo1, elmo2, self.w_post, self.gamma_post)
			elmo_post1, elmo_post2 = self.elmo_drop(elmo_post1), self.elmo_drop(elmo_post2)
			# concat again
			context = self.phi_joiner([context, elmo_post1])
			query = self.phi_joiner([query, elmo_post2])
		
		query = self.masked_fill_query(query)

		self.shared.C = context
		self.shared.Q = query

		return context, query


	def update_context(self):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		max_query_l = self.shared.query_l.max()
		hidden_size = self.opt.hidden_size

		self.char_context_view.dims = (batch_l * context_l, self.opt.token_l, self.opt.char_emb_size)
		self.char_query_view.dims = (batch_l * max_query_l, self.opt.token_l, self.opt.char_emb_size)
		self.char_context_unview.dims = (batch_l, context_l, self.opt.char_enc_size)
		self.char_query_unview.dims = (batch_l, max_query_l, self.opt.char_enc_size)


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


