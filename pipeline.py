import sys

import torch
from torch import nn
from torch import cuda
from view import *
from join_table import *
from holder import *
import numpy as np
from optimizer import *
import time

from char_embeddings import *
from embeddings import *
from encoder import *
from encoder_with_elmo import *
from biattention import *
from fused_biattention import *
from coattention import *
from reencoder import *
from match_encoder import *
from boundary_classifier import *
from boundary_chain_classifier import *
from boundary_cross_classifier import *

class Pipeline(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Pipeline, self).__init__()

		self.shared = shared
		self.opt = opt

		self.embeddings = WordVecLookup(opt, shared)
		if opt.use_char_enc == 1:
			self.char_embeddings = CharEmbeddings(opt, shared)

		# pipeline stages
		if opt.enc == 'encoder':
			self.encoder = Encoder(opt, shared)
		elif opt.enc == 'encoder_with_elmo':
			self.encoder = EncoderWithElmo(opt, shared)
		else:
			assert(False)

		if opt.att == 'biatt':
			self.attention = BiAttention(opt, shared)
		elif opt.att == 'fused_biatt':
			self.attention = FusedBiAttention(opt, shared)
		elif opt.att == 'coatt':
			self.attention = CoAttention(opt, shared)
		else:
			assert(False)

		if opt.reenc == 'reencoder':
			self.reencoder = ReEncoder(opt, shared)
		elif opt.reenc == 'match':
			self.reencoder = MatchEncoder(opt, shared)
		else:
			assert(False)

		if opt.cls == 'boundary':
			self.classifier = BoundaryClassifier(opt, shared)
		elif opt.cls == 'boundary_chain':
			self.classifier = BoundaryChainClassifier(opt, shared)
		elif opt.cls == 'boundary_cross':
			self.classifier = BoundaryCrossClassifier(opt, shared)
		else:
			assert(False)


	def init_weight(self):
		missed_names = []
		if self.opt.param_init_type == 'xavier_uniform':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_uniform_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'xavier_normal':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_normal_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'no':
			for n, p in self.named_parameters():
				missed_names.append(n)
		else:
			assert(False)

		if len(missed_names) != 0:
			print('uninitialized fields: {0}'.format(missed_names))


	def forward(self, sent1, sent2, char_sent1, char_sent2):
		shared = self.shared

		if self.opt.use_char_enc == 1:
			char_sent1 = self.char_embeddings(char_sent1)	# (batch_l, context_l, token_l, char_emb_size)
			char_sent2 = self.char_embeddings(char_sent2)	# (batch_l, query_l, token_l, char_emb_size)
		else:
			char_sent1, char_sent2 = None, None

		word_sent1 = self.embeddings(sent1)	# (batch_l, context_l, word_vec_size)
		word_sent2 = self.embeddings(sent2)	# (batch_l, query_l, word_vec_size)

		# encoder
		C, Q = self.encoder(word_sent1, word_sent2, char_sent1, char_sent2)

		# bi-attention
		att1, att2, G = self.attention(C, Q)

		# reencoder
		M = self.reencoder(G)

		# classifier
		output = self.classifier(M, G)

		return output

	# call this explicitly
	def update_context(self, batch_ex_idx, batch_l, context_l, context_sent_l, query_l, res_map=None):
		self.shared.batch_ex_idx = batch_ex_idx
		self.shared.batch_l = batch_l
		self.shared.context_l = context_l
		self.shared.context_sent_l = context_sent_l
		self.shared.query_l = query_l
		self.shared.res_map = res_map
		self.shared.max_query_l = query_l.max()
		# some constants that can be handy
		self.shared.query_mask = torch.ones(self.shared.batch_l, self.shared.max_query_l)
		for i,l in enumerate(self.shared.query_l):
			if l < self.shared.max_query_l:
				self.shared.query_mask[i, l:] = 0.0
		self.shared.query_mask = Variable(self.shared.query_mask, requires_grad=False)

		self.shared.score_mask = torch.ones(self.shared.batch_l, self.shared.context_l, self.shared.max_query_l)
		for i, l in enumerate(self.shared.query_l):
			if l < self.shared.max_query_l:
				self.shared.score_mask[i, :, l:] = 0.0
		self.shared.score_mask = Variable(self.shared.score_mask, requires_grad=False)

		self.shared.neg_inf = Variable(torch.ones(1) * -1.0e8, requires_grad=False)
		self.shared.one = Variable(torch.ones(1), requires_grad=False)
		if self.opt.gpuid != -1:
			self.shared.query_mask = self.shared.query_mask.cuda()
			self.shared.score_mask = self.shared.score_mask.cuda()
			self.shared.one = self.shared.one.cuda()
			self.shared.neg_inf = self.shared.neg_inf.cuda()


	def begin_pass(self):
		self.char_embeddings.begin_pass()
		self.embeddings.begin_pass()
		self.encoder.begin_pass()
		self.attention.begin_pass()
		self.reencoder.begin_pass()
		self.classifier.begin_pass()


	def end_pass(self):
		self.char_embeddings.end_pass()
		self.embeddings.end_pass()
		self.encoder.end_pass()
		self.attention.end_pass()
		self.reencoder.end_pass()
		self.classifier.end_pass()


	def get_param_dict(self):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		skipped_fields = []
		for n, p in self.named_parameters():
			# save all parameters that do not have skip_save flag
			# 	unlearnable parameters will also be saved
			if not hasattr(p, 'skip_save') or p.skip_save == 0:
				param_dict[n] =  torch2np(p.data, is_cuda)
			else:
				skipped_fields.append(n)
		#print('skipped fields:', skipped_fields)
		return param_dict

	def set_param_dict(self, param_dict):
		skipped_fields = []
		rec_fields = []
		for n, p in self.named_parameters():
			if n in param_dict:
				rec_fields.append(n)
				# load everything we have
				print('setting {0}'.format(n))
				p.data.copy_(torch.from_numpy(param_dict[n][:]))
			else:
				skipped_fields.append(n)
		print('skipped fileds: {0}'.format(skipped_fields))


def overfit():
	opt = Holder()
	opt.gpuid = 1
	opt.word_vec_size = 3
	opt.hidden_size = 6
	opt.dropout = 0.0
	opt.learning_rate = 0.05
	opt.birnn = 1
	opt.enc_rnn_layer = 1
	opt.reenc_rnn_layer = 2
	opt.cls_rnn_layer = 1
	opt.param_init_type = 'xavier_normal'
	shared = Holder()
	shared.batch_l = 2
	shared.context_l = 8
	shared.query_l = 5

	input1 = torch.randn(shared.batch_l, shared.context_l, opt.word_vec_size)
	input2 = torch.randn(shared.batch_l, shared.query_l, opt.word_vec_size)
	gold = torch.from_numpy(np.random.randint(shared.context_l, size=(shared.batch_l,2)))
	print('gold', gold)

	input1 = Variable(input1, True)
	input2 = Variable(input2, True)
	gold = Variable(gold, False)

	m = Pipeline(opt, shared)
	m.init_weight()

	crit1 = torch.nn.NLLLoss(size_average=False)
	crit2 = torch.nn.NLLLoss(size_average=False)

	# forward pass
	m.update_context(None, shared.batch_l, shared.context_l, shared.query_l, None)
	log_p1, log_p2 = m(input1, input2)
	print('p1', log_p1.exp())
	print('p2', log_p2.exp())

	

if __name__ == '__main__':
	overfit()
