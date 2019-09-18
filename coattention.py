import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from join_table import *
from trilinear_prod import *

# Co-attention
class CoAttention(torch.nn.Module):
	def __init__(self, opt, shared):
		super(CoAttention, self).__init__()
		self.opt = opt
		self.shared = shared

		enc_size = opt.hidden_size*2 if opt.use_elmo_post == 0 else opt.hidden_size*2 + opt.elmo_size
		self.trilinear_prod = TrilinearProd(opt, enc_size)

		self.softmax1 = nn.Softmax(1)
		self.softmax2 = nn.Softmax(2)
		self.phi_joiner = JoinTable(2)


	def coattention(self, scores, L, R):
		batch_l, left_l, enc_size = L.shape

		# attention
		att1 = self.softmax2(scores)			# (batch_l, left_l, right_l)
		att2 = self.softmax1(scores.max(2)[0])	# (batch_l, left_l)
		att2 = att2.unsqueeze(1)				# (batch_l, 1, left_l)

		# attend
		agg1 = att1.bmm(R)	# (batch_l, left_l, enc_size)
		agg2 = att2.bmm(L)	# (batch_l, 1, enc_size)
		agg2 = agg2.expand(batch_l, left_l, enc_size)
		G = self.phi_joiner([L, agg1, L * agg1, L * agg2])
		return [att1, att2, G]


	def masked_fill_scores(self, scores):
		return scores * self.shared.score_mask + (self.shared.one - self.shared.score_mask) * self.shared.neg_inf


	# input encodings of context (C) and query (Q)
	#	C of shape (batch_l, context_l, hidden_size)
	#	Q of shape (batch_l, query_l, hidden_size)
	def forward(self, C, Q):
		self.update_context()
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		max_query_l = self.shared.query_l.max()
		hidden_size = self.opt.hidden_size

		# get similarity score
		scores = self.trilinear_prod(C, Q)
		scores = self.masked_fill_scores(scores)

		#
		att1, att2, G = self.coattention(scores, C, Q)
		_, _, P = self.coattention(scores.transpose(1,2), Q, C)

		# bookkeeping
		self.shared.att_soft1 = att1
		self.shared.att_soft2 = att2
		self.shared.G = G
		self.shared.P = P

		return att1, att2, G


	def update_context(self):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		max_query_l = self.shared.query_l.max()
		word_vec_size = self.opt.word_vec_size
		hidden_size = self.opt.hidden_size

	def begin_pass(self):
		pass

	def end_pass(self):
		pass




		
