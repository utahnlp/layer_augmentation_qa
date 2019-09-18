import torch 
from torch import nn

class TrilinearProd(nn.Module):
	def __init__(self, opt, hidden_size):
		super(TrilinearProd, self).__init__()
		self.opt = opt
		self.linear1 = nn.Linear(hidden_size, 1)
		self.linear2 = nn.Linear(hidden_size, 1)
		self.linear_prod = nn.Linear(hidden_size, 1)


	# this is equivalent to "standard" trilinear product, and requires far less memory
	# x of shape (batch_l, l1, hidden_size)
	# y of shape (batch_l, l2, hidden_size)
	def forward(self, x, y):
		assert(x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2])
		batch_l, l1, hidden_size = x.shape
		l2 = y.shape[1]

		scores1 = self.linear1(x.view(-1, hidden_size)).view(batch_l, l1, 1)
		scores2 = self.linear2(y.view(-1, hidden_size)).view(batch_l, 1, l2)

		# get the weight out
		w = self.linear_prod.weight.unsqueeze(0).expand(batch_l, 1, hidden_size)
		bias = self.linear_prod.bias.unsqueeze(0).expand(batch_l, 1, 1)
		scores_prod = (w * x).bmm(y.transpose(1,2)) + bias

		return scores1 + scores2 + scores_prod
