import torch 
from torch import nn

class BilinearProd(nn.Module):
	def __init__(self, opt, hidden_size):
		super(BilinearProd, self).__init__()
		self.opt = opt
		self.W = nn.Linear(hidden_size, hidden_size, bias=False)


	# bilinear product: x W y^T
	# x of shape (batch_l, l1, hidden_size)
	# y of shape (batch_l, l2, hidden_size)
	def forward(self, x, y):
		assert(x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2])
		batch_l, l1, hidden_size = x.shape
		l2 = y.shape[1]

		W = self.W.weight.expand(batch_l, -1, -1)

		return x.bmm(W).bmm(y.transpose(1,2))
