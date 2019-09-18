import sys
sys.path.insert(0, './constraint')

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import time
from holder import *
from util import *
from a8 import *
from a9 import *

class WithinLayer(torch.nn.Module):
	def __init__(self, opt, shared):
		super(WithinLayer, self).__init__()
		self.opt = opt
		self.shared = shared
		#self.num_att_labels = opt.num_att_labels
		self.within_constr = self.get_within_constr(opt.within_constr)

		self.constr_on_att1 = False
		self.constr_on_att2 = False

		# for now, only allow constraint on attention1
		assert(self.opt.constr_on == '1')

		for t in self.opt.constr_on.split(','):
			if t == '1':
				self.constr_on_att1 = True
			elif t == '2':
				self.constr_on_att2 = True
			else:
				assert(False)

		self.zero = Variable(torch.zeros(1), requires_grad=False)
		rho_w = torch.ones(1) * opt.rho_w
		if opt.gpuid != -1:
			rho_w = rho_w.cuda()
			self.zero = self.zero.cuda()
		self.rho_w = nn.Parameter(rho_w, requires_grad=opt.fix_rho == 0)

		if len(self.within_constr) != 0:
			print('within-layer constraint enabled')


	# the function that grabs constraints
	def get_within_constr(self, names):
		layers = []
		if names == '':
			return layers
	
		for n in names.split(','):
			if n == 'a8':
				layers.append(A8(self.opt, self.shared))
			elif n == 'a9':
				layers.append(A9(self.opt, self.shared))
			else:
				print('unrecognized constraint layer name: {0}'.format(n))
				assert(False)
	
		return layers


	def forward(self, score1, score2, att1, att2):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		max_query_l = self.shared.max_query_l

		# logic pass
		batch_l = self.shared.batch_l
		datt1_ls = []
		datt2_ls = []
		for layer in self.within_constr:
			if self.constr_on_att1:
				datt1_ls.append(layer(att1).contiguous().view(1, batch_l, context_l, max_query_l))
			if self.constr_on_att2:
				assert(False)
				#datt2_ls.append(layer(att2).view(1, batch_l, max_query_l, context_l))

		datt1 = self.zero
		datt2 = self.zero
		if len(datt1_ls) != 0:
			datt1 = torch.cat(datt1_ls, 0).sum(0)
		if len(datt2_ls) != 0:
			datt2 = torch.cat(datt2_ls, 0).sum(0)
			# stats
			self.shared.w_hit_cnt = (datt2.data.sum(-1).sum(-1) > 0.0).sum()

		rho_w = self.rho_w

		constrained_score1 = score1 + rho_w * datt1
		constrained_score2 = score2 + rho_w * datt2

		# stats
		self.shared.rho_w = rho_w

		return [constrained_score1, constrained_score2]


if __name__ == '__main__':
	pass






