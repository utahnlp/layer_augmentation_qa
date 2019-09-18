import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from trilinear_prod import *
from within_layer import *

# bidir attentio
class BiAttention(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BiAttention, self).__init__()
		self.opt = opt
		self.shared = shared

		enc_size = opt.hidden_size*2 if opt.use_elmo_post == 0 else opt.hidden_size*2 + opt.elmo_size
		self.trilinear_prod = TrilinearProd(opt, enc_size)

		self.score_view = View(1,1,1,1)	# (batch_l, context_l, query_l, hidden_size)
		self.score_unview = View(1,1,1)	# (batch_l, context_l, query_l)

		if opt.within_constr != '':
			self.w_layer = WithinLayer(opt, shared)


	def biattention(self, score1, C, Q):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		enc_size = C.shape[2]

		score2 = score1.max(2)[0].unsqueeze(1)

		# attention
		att1 = nn.Softmax(2)(score1)			# (batch_l, context_l, query_l)
		att2 = nn.Softmax(2)(score2)			# (batch_l, 1, context_l)

		#
		if self.opt.within_constr != '':
			constr_score1, constr_score2 = self.w_layer(score1, score2, att1, att2)

			# recompute the attention
			att1 = nn.Softmax(2)(constr_score1)
			att2 = nn.Softmax(2)(constr_score2)

		# attend
		agg1 = att1.bmm(Q)	# (batch_l, context_l, enc_size)
		agg2 = att2.bmm(C)	# (batch_l, 1, enc_size)
		agg2 = agg2.expand(batch_l, context_l, enc_size)
		G = torch.cat([C, agg1, C * agg1, C * agg2], 2)
		return [att1, att2, G]


	def masked_fill_scores(self, scores1):
		return scores1 * self.shared.score_mask + (self.shared.one - self.shared.score_mask) * self.shared.neg_inf


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
		C = C.contiguous()
		Q = Q.contiguous()
		scores1 = self.trilinear_prod(C, Q)
		scores1 = self.masked_fill_scores(scores1)

		#
		att1, att2, G = self.biattention(scores1, C, Q)

		# bookkeeping
		self.shared.att_soft1 = att1
		self.shared.att_soft2 = att2
		self.shared.G = G

		return [att1, att2, G]


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




		
