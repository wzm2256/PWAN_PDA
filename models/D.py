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



# import torch.nn.parallel
# import torch.utils.data
# from torch.autograd import Variable
# import numpy as np
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# import pdb

# class Identity(nn.Module):
# 	def __init__(self, dim):
# 		super(Identity, self).__init__()
#
# 	def forward(self, x):
# 		return x

		# if norm_I == 0:
		# 	my_norm = Identity
		# elif norm_I == 1:
		# 	my_norm = nn.LayerNorm

# import torch
# from torch.autograd import grad
# import pdb

# Pooling each PC.
# standard network (standard pointnet)
# not symmetric h