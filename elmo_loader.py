import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *


# the elmo loader
#	it takes no input but the current example idx
#	encodings are actually loaded from cached embeddings
class ElmoLoader(torch.nn.Module):
	def __init__(self, opt, shared):
		super(ElmoLoader, self).__init__()
		self.opt = opt
		self.shared = shared


	# fetch a specific layer of elmo, 0/1/2
	def get_layer(self, idx):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		max_query_l = self.shared.max_query_l

		context = self.shared.res_map['elmo_context']
		query = self.shared.res_map['elmo_query']

		assert(query.shape[1] <= max_query_l)

		start = self.opt.elmo_in_size * idx
		end = self.opt.elmo_in_size * (idx+1)

		context = context[:, :, start:end]
		query = query[:, :, start:end]

		context = Variable(context, requires_grad=False)
		query = Variable(query, requires_grad=False)

		return context, query



	# load cached ELMo embeddings for the current batch
	#	the return tensor is of shape (3, batch_l, seq_l, elmo_in_size)
	def forward(self):
		context_l0, query_l0 = self.get_layer(0)
		context_l1, query_l1 = self.get_layer(1)
		context_l2, query_l2 = self.get_layer(2)

		return [[context_l0, context_l1, context_l2], [query_l0, query_l1, query_l2]]


	def begin_pass(self):
		pass

	def end_pass(self):
		pass

