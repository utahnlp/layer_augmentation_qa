import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from join_table import *
from trilinear_prod import *
from fusion import *

# fused bidir attention
class FusedBiAttention(torch.nn.Module):
	def __init__(self, opt, shared):
		super(FusedBiAttention, self).__init__()
		self.opt = opt
		self.shared = shared

		enc_size = opt.hidden_size if 'elmo' not in opt.enc else opt.hidden_size + opt.elmo_size
		self.trilinear_prod = TrilinearProd(opt, enc_size)

		self.fusion = Fusion(opt, enc_size)

		self.softmax2 = nn.Softmax(2)
		self.phi_joiner = JoinTable(2)


	def biattention(self, scores, C, Q):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		enc_size = C.shape[2]

		# attention
		att1 = self.softmax2(scores)					# (batch_l, context_l, max_query_l)
		att2 = self.softmax2(scores.transpose(1,2))	# (batch_l, max_query_l, context_l)

		# attend
		agg1 = att1.bmm(Q)	# (batch_l, context_l, enc_size)
		agg2 = att2.bmm(C)	# (batch_l, max_query_l, enc_size)
		agg2 = self.masked_fill_query(agg2)
		return att1, att2, agg1, agg2


	def masked_fill_scores(self, scores):
		return scores * self.shared.score_mask + (self.shared.one - self.shared.score_mask) * self.shared.neg_inf

	def masked_fill_query(self, query):
		return query * self.shared.query_mask.unsqueeze(-1)


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
		att1, att2, agg1, agg2 = self.biattention(scores, C, Q)

		#
		G = self.fusion(C, agg1)
		P = self.fusion(Q, agg2)
		P = self.masked_fill_query(P)

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




		
