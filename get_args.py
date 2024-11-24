import argparse


def get(args_ps=None):
	if args_ps is None:
		args_ps = argparse.ArgumentParser()

	### Dataset
	args_ps.add_argument('--dset', default='office_home', type=str)
	args_ps.add_argument('--worker', type=int, default=8, help="number of workers for dataloader")
	args_ps.add_argument('--batch_size', type=int, default=36)
	args_ps.add_argument('--tcls', type=int, default=0, help='number of classes, set to 0 to use default values')

	# Classifier and loss parameter
	args_ps.add_argument('--net', default='ResNet50', type=str, help='Only support ResNet50 now')
	args_ps.add_argument('--lr', type=float, default=1e-4)
	args_ps.add_argument('--cls_weight', type=float, default=1., help='classification loss weight')
	args_ps.add_argument('--entropy', type=float, default=0.1, help='entropy weight')
	args_ps.add_argument('--trade_off', default=0.05, type=float, help='PWAN loss weight')
	args_ps.add_argument('--init_fc', type=int, default=0, help='')
	args_ps.add_argument('--bottle_dim', type=int, default=256, help='Bottleneck dimension for classifer')
	args_ps.add_argument('--NoRelu', type=int, default=0, help="")

	args_ps.add_argument('--label_smooth', type=int, default=0, help="Use label smooth or not")
	args_ps.add_argument('--cot', type=int, default=0, help='Use COT or not')
	args_ps.add_argument('--cot_weight', type=float, default=0.)

	# PWAN parameter
	args_ps.add_argument('--d_leaky', default=0.2, type=float, help='leaky relu used in PWAN')
	args_ps.add_argument('--lr_D', type=float, default=1e-4)
	args_ps.add_argument('--d_weight_label', type=float, default=0.01)
	args_ps.add_argument('--d_iter', type=int, default=3, help='PWAN update frequency')
	args_ps.add_argument('--point_mass', type=float, default=9., help='set point mass manually, when use auto_ratio, this value will be ignored')
	args_ps.add_argument('--d_hidden', default=256, type=int, help='PWAN net hidden dimension size')
	args_ps.add_argument('--pm_ratio', type=float, default=1., help='point mass decrease ratio at the end of the training.')
	args_ps.add_argument('--auto_ratio', type=int, default=0, help="Estimate point mass. set to -1 to disable.")


	# Training
	args_ps.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
	args_ps.add_argument('--pre_train', type=int, default=0, help='number of pretrain step')
	args_ps.add_argument('--detach_s', type=int, default=1, help="Detach the classifer for PWAN or not")
	args_ps.add_argument('--sf', type=int, default=0, help='')
	args_ps.add_argument('--test_interval', type=int, default=100, help="interval of two continuous test phase")
	args_ps.add_argument('--visualization', type=int, default=0, help='')

	# jobs
	args_ps.add_argument('--task_list', default='0', type=str)
	args_ps.add_argument('--seed_list', default='0', type=str)

	return args_ps

def get_1(args_ps=None):
	if args_ps is None:
		args_ps = argparse.ArgumentParser()

	args_ps.add_argument('--seed', default=0, type=int)
	args_ps.add_argument('--s', type=int, default=0, help="source")
	args_ps.add_argument('--t', type=int, default=1, help="target")

	return args_ps