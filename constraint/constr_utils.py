import sys
sys.path.insert(0, '../')

import torch
from torch import nn

def activate(one, d):
	return relu(one - d)

def relu(x):
	#return nn.LeakyReLU()(x)
	return nn.ReLU()(x)

# select a > theta
def geq_theta(theta, a):
	m = (a >= theta).float()
	return a * m


def get_q_selector(opt, shared, res_name):
	# paddings are by default all 0's
	mask = torch.zeros(shared.batch_l, shared.max_query_l)

	# if res_name does not present, mark all words
	if res_name is None:
		mask[:, 1:] = 1.0
	else:
		for ex, pair in enumerate(shared.res_map[res_name]):
			q_contents = pair[1]
	
			if len(q_contents) != 0:
				mask[ex][q_contents] = 1.0
		
	if opt.gpuid != -1:
		mask = mask.cuda()
	return mask


def get_c_selector(opt, shared, res_name):
	mask = torch.zeros(shared.batch_l, shared.context_l)

	# if res_name does not present, mark all words
	if res_name is None:
		mask[:, 1:] = 1.0
	else:
		for ex, pair in enumerate(shared.res_map[res_name]):
			c_contents = pair[0]
	
			if len(c_contents) != 0:
				mask[ex][c_contents] = 1.0
		
	if opt.gpuid != -1:
		mask = mask.cuda()
	return mask


# get relation selector mask
#	the shape is according to score1 shape as (batch_l, context_l, max_query_l)
def get_rel_selector1(opt, shared, res_name):
	mask = torch.zeros(shared.batch_l, shared.context_l, shared.max_query_l)

	for ex in range(shared.batch_l):
		rel_res = shared.res_map[res_name][ex]
		rel_keys = rel_res[res_name]

		for k in rel_keys:
			src_idx, tgt_idx = rel_res[k]

			for i in src_idx:
				mask[ex][i][tgt_idx] = 1.0
	if opt.gpuid != -1:
		mask = mask.cuda()

	return mask