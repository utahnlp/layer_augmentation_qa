import torch 
from torch import nn
from torch.autograd import Variable

# basically the dynamic rnn wrapper in tensorflow
class VarRNN(nn.Module):
	def __init__(self, rnn):
		super(VarRNN, self).__init__()
		self.rnn = rnn


	def get_hidden(self, hidden, idx):
		if type(self.rnn) == nn.LSTM:
			return hidden[0][:, idx, :], hidden[1][:, idx, :]
		elif type(self.rnn) == nn.GRU:
			return hidden[:, idx, :]
		else:
			assert(False)

	
	# x is the input with variable length, size (batch_l, max_l, enc_size)
	#	where individual lengths are <= max_l
	# x_len is a long tensor of lengths,
	#	if None, x will be treated as perfectly aligned input, thus no pack/unpack is needed
	# hidden: the hidden states
	def forward(self,x, x_len, hidden=None):
		if x_len is not None:
			x_len = torch.LongTensor(x_len)
			_, pack_idx = torch.sort(-x_len, 0)
			_, unpack_idx = torch.sort(pack_idx, 0)

			# reorder input by decreasing order of lengths
			x = x[pack_idx]
			x_len = x_len[pack_idx]
			if hidden is not None:
				hidden = self.get_hidden(hidden, pack_idx)

			# pack
			x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)

			# run rnn as usual
			y, h = self.rnn(x, hidden)

			# unpack
			y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)

			# recover original input order
			y = y[unpack_idx]
			h = self.get_hidden(h, unpack_idx)
			return y, h

		else:
			return self.rnn(x, hidden)


if __name__ == '__main__':
	x = torch.randn(3,5,4)
	x[0, :5] = torch.ones(5,4)
	x[1, :2] = torch.ones(2,4)
	x[2, :3] = torch.ones(3,4)
	x_len = [5,2,3]
	x = Variable(x)

	rnn = VarRNN(nn.GRU(4, 1, batch_first=True, bidirectional=True))

	y, h = rnn(x, x_len)

	print(y)
	print(h)

