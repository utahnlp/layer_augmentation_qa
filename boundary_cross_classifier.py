import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from join_table import *
from locked_dropout import *
from bilinear_prod import *
from self_attention import *

# boundary cross classifier
class BoundaryCrossClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BoundaryCrossClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		self.selfatt_p = SelfAttention(opt, shared, opt.hidden_size, prod_type='linear', mask_type='query')

		self.bilinear_prod1 = BilinearProd(opt, opt.hidden_size)
		self.bilinear_prod2 = BilinearProd(opt, opt.hidden_size)

		self.drop = LockedDropout(opt.dropout)

		self.linear_view = View(1,1,1)
		self.linear_unview = View(1,1)

		bidir = opt.birnn == 1
		rnn1_in_size = opt.hidden_size * 1
		rnn2_in_size = opt.hidden_size * 1
		rnn_hidden_size = opt.hidden_size if not bidir else opt.hidden_size//2
		self.rnn1 = build_rnn(
			opt.rnn_type,
			input_size=rnn1_in_size,
			hidden_size=rnn_hidden_size,
			num_layers=opt.cls_rnn_layer,
			bias=True,
			batch_first=True,
			dropout=opt.dropout,
			bidirectional=bidir)
		self.rnn2 = build_rnn(
			opt.rnn_type,
			input_size=rnn2_in_size,
			hidden_size=rnn_hidden_size,
			num_layers=opt.cls_rnn_layer,
			bias=True,
			batch_first=True,
			dropout=opt.dropout,
			bidirectional=bidir)

		self.logsoftmax = nn.LogSoftmax(1)
		self.phi_joiner = JoinTable(2)


	def rnn_over(self, rnn, x):
		x , _ = rnn(self.drop(x))
		x = self.drop(x)
		return x


	# G: output of biattention of shape (batch_l, context_l, enc_size1)
	# M: output of match_encoder of shape (batch_l, context_l, enc_size2)
	def forward(self, M, G):
		self.update_context()

		q = self.self_attention_p(P)	# (batch_l, 1, hidden_size)
		assert(q.shape == (self.shared.batch_l, 1, self.opt.hidden_size))

		M1 = M
		phi1 = self.rnn_over(self.rnn1, M1).contiguous()
		y_scores1 = self.bilinear_prod1(q, phi1).squeeze(1)

		M2 = self.phi_joiner([phi1, M1])
		phi2 = self.rnn_over(self.rnn2, M2).contiguous()
		y_scores2 = self.bilinear_prod2(q, phi2).squeeze(1)

		# log probabilities
		log_p1 = self.logsoftmax(y_scores1)
		log_p2 = self.logsoftmax(y_scores2)

		return log_p1, log_p2


	def update_context(self):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		hidden_size = self.opt.hidden_size


	def begin_pass(self):
		pass

	def end_pass(self):
		pass



