import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from join_table import *


class CharRNN(torch.nn.Module):
	def __init__(self, opt, shared):
		super(CharRNN, self).__init__()
		self.opt = opt
		self.shared = shared

		# input dim char_emb_size
		# output dim hidden_size/2
		self.bidir = opt.birnn == 1
		rnn_in_size = opt.char_emb_size
		rnn_hidden_size = opt.hidden_size/2 if not self.bidir else opt.hidden_size/4

		if opt.rnn_type == 'lstm':
			self.rnn = nn.LSTM(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=1,
				bias=True,
				batch_first=True,
				dropout=opt.dropout_h,
				bidirectional=self.bidir)

		elif opt.rnn_type == 'gru':
			self.rnn = nn.GRU(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=1,
				bias=True,
				batch_first=True,
				dropout=opt.dropout_h,
				bidirectional=self.bidir)
		else:
			assert(False)


	# input size (batch_l * seq_l, token_l, char_emb_size)
	# output size (batch_l * seq_l, hidden_size/2)
	def forward(self, x):
		num_tok, token_l, char_emb_size = x.shape

		char_enc, (h, _) = self.rnn(x)	# (batch_l * seq_l, token_l, hidden_size/2)

		tok_enc = h.transpose(0,1).contiguous().view(num_tok, self.opt.hidden_size/2)

		return tok_enc




if __name__ == '__main__':
	#batch_l = 2
	#num_kernel = 3
	#emb_size = 4
	#seq_l = 4
	#filter_size = 3
	#input_channel = 1
	#a = Variable(torch.ones(batch_l, seq_l, emb_size))
	#a = a.unsqueeze(1)
	#conv = nn.Conv2d(input_channel, num_kernel, (filter_size, emb_size))
#
	#print('a', a)
	#print('conv', conv)
#
	#
	#out = conv(a)
	#print('out', out)
#
	#out = out.squeeze(-1)
#
	#max_out = nn.MaxPool1d(out.size(2))(out)
	#print('max_out', max_out)



	shared = Holder()
	opt = Holder()
	opt.token_l = 20
	opt.batch_l = 2
	opt.hidden_size = 200
	opt.char_emb_size = 16
	opt.filter_sizes = '5'
	opt.dropout = 0.0

	conv = CharCNN(opt, shared)
	a = Variable(torch.randn(opt.batch_l, opt.token_l, opt.char_emb_size))
	print('a', a)
	print(a.shape)

	out = conv(a)
	print('out', out)
	print(out.shape)

