import torch
from torch.autograd import grad
import pdb


class Grad_Penalty:

	def __init__(self, lambdaGP, point_mass, gamma=1, device=torch.device('cpu'), ):
		self.lambdaGP = lambdaGP
		self.gamma = gamma
		self.device = device
		self.point_mass = point_mass

	def __call__(self, loss, All_points):

		gradients = grad(outputs=loss, inputs=[i.contiguous() for i in All_points],
						 grad_outputs=torch.ones(loss.size()).to(self.device).contiguous(),
						 create_graph=True, retain_graph=True)

		grad_all = torch.cat(gradients, 0)
		# pdb.set_trace()
		gradient_penalty = (torch.nn.functional.relu(grad_all.norm(2, dim=1) - self.gamma) ** 2).mean() * self.lambdaGP

		with torch.no_grad():
			# grad_norm = grad_all.norm(2, dim=1, keepdim=True)
			M_grad = torch.max(grad_all.norm(2, dim=1))

		return gradient_penalty, M_grad



def cal_dloss(potential1, potential2, point_mass):
	D_real_int = torch.mean(potential1)
	D_fake_int = torch.mean(potential2 * point_mass)
	d_loss = D_fake_int - D_real_int
	return d_loss

def Concate_w(f, l, weight=1.):
	out = torch.cat([f, weight * torch.nn.functional.softmax(l, dim=-1)], -1)
	return out


def Entropy(logits):
	# logits BxN
	min_real = torch.finfo(logits.dtype).min
	logits = torch.clamp(logits, min=min_real)
	logits = logits - logits.logsumexp(dim=-1, keepdim=True)
	probs = torch.exp(logits)
	p_log_p = logits * probs

	return -p_log_p.sum(-1)