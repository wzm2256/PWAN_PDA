import torch
import torch.nn as nn

class D2(nn.Module):
	def __init__(self, in_feature, hidden_size, leaky=0.2):
		super(D2, self).__init__()

		self.Process = nn.Sequential(
			nn.Linear(in_feature, hidden_size),
			nn.LeakyReLU(negative_slope=leaky),
			nn.Linear(hidden_size, hidden_size),
			nn.LeakyReLU(negative_slope=leaky),
			nn.Linear(hidden_size, 1),
		)

	def forward(self, x):

		p_out = self.Process(x)
		out = -torch.abs(p_out)

		return out