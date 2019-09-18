
import sys
sys.path.append('../allennlp')
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from locked_dropout import *
from allennlp.modules.elmo import Elmo, batch_to_ids

############## NOT YET WORKING
# the elmo scanner
# the linear summation, gamma scaling, and elmo dropout are included
class ElmoEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(ElmoEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		self.num_output = 2 if opt.use_elmo_post == 1 else 1

		# initialize from these
		options_file = None
		weight_file = None
		if opt.elmo_in_size == 1024:
			options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
			weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
		elif opt.elmo_in_size == 512:
			options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
			weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

		self.elmo = Elmo(options_file, weight_file, num_output_representations=self.num_output, dropout=opt.elmo_dropout, requires_grad=opt.fix_elmo == 0)

		# skip initialization
		for n, p in self.elmo.named_parameters():
			p.skip_init = True


	# a simple heuristics to get batch size to keep ELMo model memory under control (~ 2GB)
	def get_batch_size(self, length):
		return round(900.0 / length * 3)


	# elmo scan over batch of sequences (tokenized)
	#	return tensor of shape (num_output, batch_l, seq_l, elmo_in_size)
	def elmo_over(self, toks):
		elmo_batch_size = self.get_batch_size(len(toks))
		rs = []
		for i in range(0, len(toks), elmo_batch_size):
			char_idx = batch_to_ids(toks[i:i+elmo_batch_size])
			if self.opt.gpuid != -1:
				char_idx = char_idx.cuda()
			emb = self.elmo(char_idx)['elmo_representations']	# list of num_output (elmo_batch_size, seq_l, elmo_in_size)

			emb = torch.cat([t.unsqueeze(0) for t in emb], 0)	# (num_output, elmo_batch_size, seq_l, elmo_in_size)
			rs.append(emb)
		return torch.cat(rs, 1)


	def forward(self):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		max_query_l = self.shared.query_l.max()
		elmo_batch_size = self.get_batch_size(context_l)

		context_toks = self.shared.res_map['context']
		query_toks = self.shared.res_map['query']

		assert(batch_l == len(context_toks))

		elmo1 = self.elmo_over(context_toks)
		elmo2 = self.elmo_over(query_toks)

		print(elmo1.shape)

		#assert(emb1.is_cuda is False)
		assert(elmo1.shape == (self.num_output, batch_l, context_l, self.opt.elmo_in_size))
		assert(elmo2.shape == (self.num_output, batch_l, max_query_l, self.opt.elmo_in_size))

		return elmo1, elmo2
