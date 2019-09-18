import torch 
from torch import nn

class LinearProd(nn.Module):
	def __init__(self, opt, hidden_size):
		super(LinearProd, self).__init__()
		self.opt = opt
		self.w = nn.Linear(hidden_size, 1, bias=True)


	# linear product: \sigma(w^T x) -> a
	#	return a \cdot x
	def forward(self, x):
		batch_l, seq_l, hidden_size = x.shape
		scores = self.w(x.view(-1, hidden_size)).view(batch_l, 1, seq_l)
		return scores
