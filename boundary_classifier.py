import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from join_table import *
from dropout_lstm import *
from locked_dropout import *

# boundary classifier
class BoundaryClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BoundaryClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		# weights will be initialized later
		self.w_p1 = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*10, 1))

		self.w_p2 = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*10, 1))

		# dropout for rnn
		self.drop = nn.Dropout(opt.dropout)

		# temp shape what will be adjusted on the fly
		batch_l = 1
		context_l = 2
		self.phi_view = View(batch_l * context_l, opt.hidden_size*10)
		self.phi_unview = View(batch_l, context_l)

		# rnn for m2
		self.bidir = opt.birnn == 1
		rnn_in_size = opt.hidden_size*2 * 7
		rnn_hidden_size = opt.hidden_size*2 if not self.bidir else opt.hidden_size
		if opt.rnn_type == 'lstm':
			self.rnn = nn.LSTM(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=opt.cls_rnn_layer,
				bias=True,
				batch_first=True,
				dropout=opt.dropout,
				bidirectional=self.bidir)
		elif opt.rnn_type == 'gru':
			self.rnn = nn.GRU(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=opt.cls_rnn_layer,
				bias=True,
				batch_first=True,
				dropout=opt.dropout,
				bidirectional=self.bidir)
		else:
			assert(False)

		self.logsoftmax = nn.LogSoftmax(1)
		self.softmax1 = nn.Softmax(1)
		self.phi_joiner = JoinTable(2)


	def rnn_over(self, context):
		if self.opt.rnn_type == 'lstm' or self.opt.rnn_type == 'gru':
			M2, _ = self.rnn(self.drop(context))
			return M2
		else:
			assert(False)


	# input aggregation encodings and re-encodings
	#	M: re-encodings of size (batch_l, context_l, hidden_size)
	#	G: aggregation encodings of size (batch_l, context_l, hidden_size * 4)
	# NOTENOTENOTE:
	#	The code becomes diff from the formulation in arxiv paper (according to bidaf code)
	def forward(self, M, G):
		self.update_context()

		# start index
		phi1 = self.phi_joiner([G, M])				# (batch_l, context_l, hidden_size*5)
		y_scores1 = self.phi_unview(self.w_p1(self.phi_view(phi1)))		# (batch_l, context_l)
		p1 = self.softmax1(y_scores1).unsqueeze(1)	# (batch_l, 1, context_l)

		# end index
		A = p1.bmm(M).expand(-1, self.shared.context_l, -1)	# (batch_l, context_l, hidden_size)
		G2 = self.phi_joiner([G, M, A, M * A])		# (batch_l, context_l, hidden_size * 7)

		M2 = self.rnn_over(G2)					# (batch_l, context_l, hidden_size)
		phi2 = self.phi_joiner([G, M2])				# (batch_l, context_l, hidden_size*5)
		y_scores2 = self.phi_unview(self.w_p2(self.phi_view(phi2)))		# (batch_l, context_l)

		# log softmax
		log_p1 = self.logsoftmax(y_scores1)
		log_p2 = self.logsoftmax(y_scores2)

		return [log_p1, log_p2]


	def update_context(self):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		hidden_size = self.opt.hidden_size

		self.phi_view.dims = (batch_l * context_l, hidden_size * 10)
		self.phi_unview.dims = (batch_l, context_l)


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





		
