import torch

def cal_dloss(potential1, potential2, point_mass):
	D_real_int = torch.mean(potential1)
	D_fake_int = torch.mean(potential2 * point_mass)
	d_loss = D_fake_int - D_real_int
	return d_loss

def Concate_w(f, l, weight=1.):
	out = torch.cat([f, weight * l], -1)
	return out


def Entropy(logits):
	# logits BxN
	min_real = torch.finfo(logits.dtype).min
	logits = torch.clamp(logits, min=min_real)
	logits = logits - logits.logsumexp(dim=-1, keepdim=True)
	probs = torch.exp(logits)
	p_log_p = logits * probs

	return -p_log_p.sum(-1)