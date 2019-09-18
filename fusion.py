import torch 
from torch import nn
from torch.autograd import Variable

class Fusion(nn.Module):
	def __init__(self, opt, hidden_size):
		super(Fusion, self).__init__()
		self.opt = opt
		self.match = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(hidden_size * 4, hidden_size),
			nn.Tanh())
		self.gate = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(hidden_size * 4, 1),
			nn.Sigmoid())


	# x of shape (batch_l, l1, hidden_size)
	# y of shape (batch_l, l2, hidden_size)
	# t = tanh(w1^T [x, y, x-y, x*y])
	# g = sigm(w2^T [x, y, x-y, x*y])
	# t * g + (1-g) * x
	def forward(self, x, y):
		assert(x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2])
		batch_l, l1, hidden_size = x.shape
		l2 = y.shape[1]

		one = Variable(torch.ones(1), requires_grad=False)
		if self.opt.gpuid != -1:
			one = one.cuda()

		merged = torch.cat([x, y, x-y, x*y], 2)
		matched = self.match(merged)
		gated = self.gate(merged)

		return gated * matched + (one - gated) * x
