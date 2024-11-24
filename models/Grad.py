import torch
from torch.autograd import grad
import pdb


class Grad_Penalty_w:

	def __init__(self, lambdaGP, gamma=1):
		self.lambdaGP = lambdaGP
		self.gamma = gamma

	def __call__(self, loss, All_points, ratio):

		bs = All_points[0].shape[0]
		gradients = grad(outputs=loss, inputs=[i.contiguous() for i in All_points],
						 grad_outputs=torch.ones(loss.size()).to(All_points[0].device).contiguous(),
						 create_graph=True, retain_graph=True)

		grad_all = torch.cat([gradients[0] * ratio * bs, gradients[1] * bs], 0)

		source_norm = grad_all.norm(2, dim=1)[:All_points[0].shape[0]]
		all_norm = grad_all.norm(2, dim=1)

		gradient_penalty = (torch.nn.functional.relu(grad_all.norm(2, dim=1) - self.gamma) ** 2).mean() * self.lambdaGP

		with torch.no_grad():
			M_grad = torch.max(grad_all.norm(2, dim=1))

		return gradient_penalty, M_grad, source_norm.detach(), all_norm.detach(), grad_all
