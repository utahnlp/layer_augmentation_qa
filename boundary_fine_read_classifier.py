import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from join_table import *
from locked_dropout import *
from bilinear_prod import *
from var_rnn import *
from fusion import *

# boundary fine read classifier
class BoundaryFineReadClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BoundaryFineReadClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		self.linear1 = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*4, 1))

		self.linear2 = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*4, 1))

		self.linear_view = View(1,1,1)
		self.linear_unview = View(1,1)

		self.drop = LockedDropout(opt.dropout)

		bidir = opt.birnn == 1
		rnn1_in_size = opt.hidden_size * 2
		rnn2_in_size = opt.hidden_size * 4
		rnn_hidden_size = opt.hidden_size*2 if not bidir else opt.hidden_size
		self.rnn1 = VarRNN(build_rnn(
			opt.rnn_type,
			input_size=rnn1_in_size,
			hidden_size=rnn_hidden_size,
			num_layers=opt.cls_rnn_layer,
			bias=True,
			batch_first=True,
			dropout=opt.dropout,
			bidirectional=bidir))
		self.rnn2 = VarRNN(build_rnn(
			opt.rnn_type,
			input_size=rnn2_in_size,
			hidden_size=rnn_hidden_size,
			num_layers=opt.cls_rnn_layer,
			bias=True,
			batch_first=True,
			dropout=opt.dropout,
			bidirectional=bidir))

		self.logsoftmax = nn.LogSoftmax(1)
		self.phi_joiner = JoinTable(2)
		self.sent_joiner = JoinTable(1)


	def rnn_over(self, rnn, x, x_len, hidden):
		#x = self.drop(x)
		x, h = rnn(x, x_len, hidden)
		return x, h


	def rnn_over_sent(self, rnn, x, x_sent_len):
		acc_l = 0
		sents = []
		for l in x_sent_len:
			s, _ = rnn(x[:, acc_l:acc_l+l, :], None, None)
			acc_l += l
			sents.append(s)
		return self.sent_joiner(sents), None


	# G: output of biattention of shape (batch_l, context_l, enc_size1)
	# M: output of match_encoder of shape (batch_l, context_l, enc_size2)
	def forward(self, M, G):
		assert(self.shared.context_sent_l[0] == self.shared.context_sent_l[-1])
		context_sent_l = self.shared.context_sent_l[0]
		self.update_context()

		M = self.drop(M)

		M1_all, _ = self.rnn_over(self.rnn1, M, None, None)
		M1_sent, _ = self.rnn_over_sent(self.rnn1, M, context_sent_l)
		phi1 = self.phi_joiner([M1_all, M1_sent])
		y_scores1 = self.linear_unview(self.linear1(self.linear_view(phi1)))

		M1_all = self.drop(M1_all)

		M2 = self.phi_joiner([M, M1_all])
		M2_all, _ = self.rnn_over(self.rnn2, M2, None, None)
		M2_sent, _ = self.rnn_over_sent(self.rnn2, M2, context_sent_l)
		phi2 = self.phi_joiner([M2_all, M2_sent])
		y_scores2 = self.linear_unview(self.linear2(self.linear_view(phi2)))

		# log probabilities
		log_p1 = self.logsoftmax(y_scores1)
		log_p2 = self.logsoftmax(y_scores2)

		return log_p1, log_p2


	def update_context(self):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		hidden_size = self.opt.hidden_size

		self.linear_view.dims = (batch_l * context_l, hidden_size*4)
		self.linear_unview.dims = (batch_l, context_l)


	def begin_pass(self):
		pass

	def end_pass(self):
		pass



