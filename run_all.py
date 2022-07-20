import os
import subprocess
import argparse
import numpy as np
import pdb
import prepare_data

args_ps = argparse.ArgumentParser()


args_ps.add_argument('--dset', default='office_home', type=str)
args_ps.add_argument('--trade_off', default=0.1, type=float)
args_ps.add_argument('--d_hidden', default=256, type=int)
args_ps.add_argument('--d_norm', default=0, type=int)
args_ps.add_argument('--d_leaky', default=0.2, type=float)
args_ps.add_argument('--lr_D', type=float, default=1e-4)
args_ps.add_argument('--lr_G', type=float, default=1e-4)
args_ps.add_argument('--d_weight_label', type=float, default=0.01)
args_ps.add_argument('--d_iter', type=int, default=3)
args_ps.add_argument('--point_mass', type=float, default=0.25)
args_ps.add_argument('--cls_weight', type=float, default=1.)
args_ps.add_argument('--opt_G', type=int, default=3, help='0: SGD momentum 1:RMSprop 2: SGD ')
args_ps.add_argument('--opt_D', type=int, default=0, help='0: SGD momentum 1:RMSprop 2: SGD ')
args_ps.add_argument('--entropy', type=float, default=0.1, help='entropy weight')
args_ps.add_argument('--entropy_s', type=float, default=0., help='entropy weight')
args_ps.add_argument('--batch_size', type=int, default=36)
args_ps.add_argument('--worker', type=int, default=8, help="number of workers")
args_ps.add_argument('--pre_process', type=int, default=0, help='')
args_ps.add_argument('--init_fc', type=int, default=0, help='')
args_ps.add_argument('--pm_ratio', type=float, default=1., help='point mass decrease ratio at the end of the training.')
args_ps.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
args_ps.add_argument('--mass_inc', default=0, type=int)
args_ps.add_argument('--seed', default='0', type=str)
args_ps.add_argument('--entropy_w', type=float, default=0, help="")


args_ps.add_argument('--auto_ratio', type=int, default=0, help="")
args_ps.add_argument('--detach_ent', type=int, default=0, help="")
args_ps.add_argument('--label_smooth', type=int, default=0, help="")
args_ps.add_argument('--cat_smooth', type=int, default=0, help="")
args_ps.add_argument('--NoRelu', type=int, default=0, help="")
args_ps.add_argument('--normalize', type=int, default=0, help="")
args_ps.add_argument('--pre_train', type=int, default=0, help='')
args_ps.add_argument('--cot', type=int, default=0, help='0: no, 1:uniform weight, 2: use weight')
args_ps.add_argument('--cot_weight', type=float, default=0., help='')
args_ps.add_argument('--t_cls', type=int, default=0, help='')
args_ps.add_argument('--t_cls_weight', type=float, default=0., help='')
args_ps.add_argument('--clsw', type=int, default=0, help='')

args_ps.add_argument('--task_list', default='0', type=str)

args = args_ps.parse_args()

def run(repeat_time, ex_str, ex_value):
	Setting = ''
	for l in range(len(ex_str)):
		Setting += ' '
		Setting += ex_str[l]
		Setting += ' '
		Setting += str(ex_value[l])

	Default_cmd ='python PWANN.py   --net ResNet50 '
	cmd = Default_cmd + Setting

	print(cmd)

	for r in range(repeat_time):
		subprocess.run(cmd, shell=True)

########## Office31 all_task
ST = [ 
    (0,1),
    (0,2),
    (0,3),
    (1,0),
    (1,2),
    (1,3),
    (2,0),
    (2,1),
    (2,3),
    (3,0),
    (3,1),
    (3,2),
]

task_list = [int(i) for i in args.task_list.strip().split(',')]
seed_list = [int(i) for i in args.seed.strip().split(',')]

# prepare_data.OfficeHome(root='data/office_home/images/')

for task in task_list:
	st = ST[task]
	for s in seed_list:
		run(1, ['--dset', '--s', '--t', '--max_iterations', '--batch_size', '--worker', '--d_leaky',
		                         '--point_mass', '--trade_off', '--d_iter', '--d_norm',
		                         '--d_hidden', '--lr_D', '--d_weight_label', '--cls_weight',
		                         '--opt_G', '--entropy', '--entropy_s', '--lr',
		                        '--pre_process', '--init_fc', '--pm_ratio', '--opt_D',
		                        '--mass_inc', '--seed', '--auto_ratio', '--detach_ent',
		                        '--label_smooth', '--cat_smooth', '--NoRelu', '--normalize',
		                        '--entropy_w', '--pre_train', '--cot', '--cot_weight',
		                        '--t_cls', '--t_cls_weight', '--clsw'],
			[args.dset, st[0], st[1], args.max_iterations, args.batch_size, args.worker, args.d_leaky,
			 args.point_mass, args.trade_off, args.d_iter, args.d_norm,
			 args.d_hidden, args.lr_D, args.d_weight_label, args.cls_weight,
			 args.opt_G, args.entropy, args.entropy_s, args.lr_G,
			  args.pre_process, args.init_fc, args.pm_ratio, args.opt_D,
			 args.mass_inc, s, args.auto_ratio, args.detach_ent,
			 args.label_smooth, args.cat_smooth, args.NoRelu, args.normalize,
			 args.entropy_w, args.pre_train, args.cot, args.cot_weight,
			 args.t_cls, args.t_cls_weight, args.clsw])