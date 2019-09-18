import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from join_table import *
from trilinear_prod import *
from bilinear_prod import *
from linear_prod import *

# self attention
class SelfAttention(torch.nn.Module):
	def __init__(self, opt, shared, hidden_size, prod_type, mask_type):
		super(SelfAttention, self).__init__()
		self.opt = opt
		self.shared = shared
		self.prod_type = prod_type
		self.mask_type = mask_type

		if self.prod_type == 'linear':
			self.prod = LinearProd(opt, hidden_size)
		elif self.prod_type == 'bilinear':
			self.prod = BilinearProd(opt, hidden_size)
		elif self.prod_type == 'trilinear' or self.prod_type == 'trilinear_norm':
			self.prod = TrilinearProd(opt, hidden_size)
			self.bias = nn.Parameter(-torch.ones(1), requires_grad=True)
			self.bias.skip_init = True

		self.phi_joiner = JoinTable(2)
		self.softmax = nn.Softmax(2)


	def normalize(self, scores):
		if self.prod_type == 'linear' or self.prod_type == 'bilinear':
			return self.softmax(scores)
		elif self.prod_type == 'trilinear' or self.prod_type == 'trilinear_norm':
			exp = scores.exp()
			p = exp / (exp.sum(2, keepdim=True) + self.bias.exp())
			return p
		assert(False)


	def masked_fill_diagonal(self, scores):
		mask = Variable(torch.eye(scores.shape[1]), requires_grad=False).unsqueeze(0)
		if self.opt.gpuid != -1:
			mask = mask.cuda()

		return self.shared.neg_inf * mask + (self.shared.one - mask) * scores


	def get_similarity(self, x):
		if self.prod_type == 'linear':
			return self.prod(x)
		elif self.prod_type == 'bilinear':
			return self.prod(x, x)
		elif self.prod_type == 'trilinear':
			return self.prod(x, x)
		elif self.prod_type == 'trilinear_norm':
			scores = self.prod(x, x)	# scale it as "att is all you need"
			scale = torch.ones(1) * (1.0 / np.sqrt(x.shape[1]))
			if x.is_cuda:
				scale = scale.cuda()
			return scores * scale
		assert(False)


	def masked_fill_scores(self, scores):
		if self.mask_type == 'diagonal':
			return self.masked_fill_diagonal(scores)
		elif self.mask_type == 'query':
			if self.prod_type == 'linear':
				mask = self.shared.query_mask.unsqueeze(1)
				return scores * mask + (self.shared.one - mask) * self.shared.neg_inf
			else:
				scores = self.masked_fill_diagonal(scores)
				mask = self.shared.query_mask.unsqueeze(-1).bmm(self.shared.query_mask.unsqueeze(1))
				return scores * mask + (self.shared.one - mask) * self.shared.neg_inf
		assert(False)


	def forward(self, x):
		x = x.contiguous()

		# get similarity scores
		scores = self.get_similarity(x)

		# block diagonal alignment
		if self.mask_type is not None:
			scores = self.masked_fill_scores(scores)

		# get normalized alignemnt
		p = self.normalize(scores)

		# attend
		agg = p.bmm(x)

		# in case x is query and the prod type is not linear,
		#	mask the result further
		if self.mask_type == 'query' and self.prod_type != 'linear':
			agg = p.bmm(x) * self.shared.query_mask.unsqueeze(-1)
			
		return agg


