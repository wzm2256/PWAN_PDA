import argparse


def get(args_ps=None):
	if args_ps is None:
		args_ps = argparse.ArgumentParser()

	# to remove
	args_ps.add_argument('--opt_G', type=int, default=3, help='0: SGD momentum 1:RMSprop 2: SGD ')
	args_ps.add_argument('--opt_D', type=int, default=0, help='0: SGD momentum 1:RMSprop 2: SGD ')
	args_ps.add_argument('--cat_smooth', type=int, default=0, help="")
	args_ps.add_argument('--pre_process', type=int, default=0, help='')
	args_ps.add_argument('--skip_first', type=int, default=0, help="")
	args_ps.add_argument('--normalize', type=int, default=0, help="")
	args_ps.add_argument('--leaky', type=float, default=0., help='')
	args_ps.add_argument('--WG', type=int, default=0, help='')
	args_ps.add_argument('--d_norm', default=0, type=int)
	args_ps.add_argument('--clsw', type=int, default=2, help='')
	args_ps.add_argument('--t_cls', type=int, default=0, help='')
	args_ps.add_argument('--t_cls_weight', type=float, default=0., help='')
	args_ps.add_argument('--warm', type=int, default=0, help='')
	args_ps.add_argument('--detach_ent', type=int, default=0, help="")
	###

	args_ps.add_argument('--net', default='ResNet50', type=str)
	args_ps.add_argument('--dset', default='office_home', type=str)
	args_ps.add_argument('--trade_off', default=0.1, type=float)
	args_ps.add_argument('--d_hidden', default=256, type=int)

	args_ps.add_argument('--d_leaky', default=0.2, type=float)
	args_ps.add_argument('--lr_D', type=float, default=1e-4)
	args_ps.add_argument('--lr', type=float, default=1e-4)
	args_ps.add_argument('--d_weight_label', type=float, default=0.01)
	args_ps.add_argument('--d_iter', type=int, default=3)
	args_ps.add_argument('--point_mass', type=float, default=0.25)
	args_ps.add_argument('--cls_weight', type=float, default=1.)
	args_ps.add_argument('--entropy', type=float, default=0.1, help='entropy weight')
	args_ps.add_argument('--entropy_s', type=float, default=0., help='entropy weight')
	args_ps.add_argument('--batch_size', type=int, default=36)
	args_ps.add_argument('--worker', type=int, default=8, help="number of workers")



	args_ps.add_argument('--init_fc', type=int, default=0, help='')
	args_ps.add_argument('--pm_ratio', type=float, default=1., help='point mass decrease ratio at the end of the training.')
	args_ps.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
	args_ps.add_argument('--mass_inc', default=0, type=int)

	args_ps.add_argument('--entropy_w', type=float, default=0, help="")
	args_ps.add_argument('--detach_s', type=int, default=0, help="")
	args_ps.add_argument('--auto_ratio', type=int, default=0, help="")

	args_ps.add_argument('--label_smooth', type=int, default=0, help="")
	args_ps.add_argument('--NoRelu', type=int, default=0, help="")

	args_ps.add_argument('--pre_train', type=int, default=0, help='')
	args_ps.add_argument('--cot', type=int, default=0, help='0: no, 1:uniform weight, 2: use weight')
	args_ps.add_argument('--cot_weight', type=float, default=0., help='')



	args_ps.add_argument('--bottle_dim', type=int, default=256, help='')
	args_ps.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")

	args_ps.add_argument('--tcls', type=int, default=0, help='')
	args_ps.add_argument('--sf', type=int, default=0, help='')
	args_ps.add_argument('--visualization', type=int, default=1, help='')

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




# def get1(args_ps=None):
# 	parser = argparse.ArgumentParser(description='BA3US for Partial Domain Adaptation')
#
#
#
# 	# new
# 	parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
#
#
# 	# share
# 	parser.add_argument('--dset', type=str, default='OfficeHome',
# 	                choices=["VisDA2017", "OfficeHome", "ImageNetCaltech", "DomainNet"])
# 	parser.add_argument('--trade_off', default=1., type=float,
# 	                help='the trade-off hyper-parameter for transfer loss')
#
# 	parser.add_argument('--d_hidden', default=256, type=int)
# 	parser.add_argument('--d_norm', default=0, type=int)
# 	parser.add_argument('--d_leaky', default=0.2, type=float)
# 	parser.add_argument('--lr_D', type=float, default=1e-4)
# 	parser.add_argument('--d_weight_label', type=float, default=0.01)
# 	parser.add_argument('--d_iter', type=int, default=2)
# 	parser.add_argument('--point_mass', type=float, default=1.0)
# 	parser.add_argument('--cls_weight', type=float, default=1.)
# 	parser.add_argument('--t_cls_weight', type=float, default=0., help='')
# 	parser.add_argument('--opt_G', type=int, default=0, help='0: SGD momentum 1:RMSprop 2: SGD 3: RMSProp no schedule')
# 	parser.add_argument('--opt_D', type=int, default=0, help='0: SGD momentum 1:RMSprop 2: SGD 3: RMSProp no schedule')
# 	parser.add_argument('--entropy', type=float, default=0., help='entropy weight')
# 	parser.add_argument('--entropy_s', type=float, default=0., help='entropy weight')
# 	parser.add_argument('--entropy_w', type=float, default=0, help="")
# 	parser.add_argument('--batch_size', type=int, default=65, help="batch_size")
# 	parser.add_argument('--worker', type=int, default=4, help="number of workers")
# 	parser.add_argument('--pre_process', type=int, default=0, help='')
#
#
#
# 	######
#
#
#
# 	parser.add_argument('--seed', type=int, default=0, help="random seed")
# 	parser.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
# 	parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet50", "VGG16"])
#
#
# 	parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
# 	parser.add_argument('--auto_ratio', type=int, default=0, help="")
# 	parser.add_argument('--detach_ent', type=int, default=0, help="")
# 	parser.add_argument('--label_smooth', type=int, default=0, help="")
# 	parser.add_argument('--cat_smooth', type=int, default=0, help="")
# 	parser.add_argument('--NoRelu', type=int, default=0, help="")
# 	parser.add_argument('--normalize', type=int, default=0, help="")
# 	parser.add_argument('--detach_s', type=int, default=0, help="")
# 	parser.add_argument('--skip_first', type=int, default=0, help="")
#
# 	########
# 	parser.add_argument('--mass_inc', default=0, type=int)
#
#
#
#
#
#
# 	parser.add_argument('--init_fc', type=int, default=0, help='')
# 	parser.add_argument('--pre_train', type=int, default=0, help='')
# 	parser.add_argument('--cot', type=int, default=0, help='0: no, 1:uniform weight, 2: use weight')
# 	parser.add_argument('--cot_weight', type=float, default=0., help='')
# 	parser.add_argument('--t_cls', type=int, default=0, help='')
# 	parser.add_argument('--clsw', type=int, default=0, help='')
# 	parser.add_argument('--bottle_dim', type=int, default=256, help='')
# 	parser.add_argument('--tcls', type=int, default=0, help='')
# 	parser.add_argument('--sf', type=int, default=0, help='')
# 	parser.add_argument('--warm', type=int, default=0, help='')
# 	parser.add_argument('--WG', type=int, default=0, help='')
# 	parser.add_argument('--leaky', type=float, default=0., help='')
#
# 	parser.add_argument('--pm_ratio', type=float, default=1., help='point mass decrease ratio at the end of the training.')
#
# 	return parser