import os
import subprocess
import argparse
import numpy as np
import pdb
import prepare_data

args_ps = argparse.ArgumentParser()


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
args_ps.add_argument('--entropy', type=float, default=0.1, help='entropy weight')
args_ps.add_argument('--entropy_s', type=float, default=0., help='entropy weight')
args_ps.add_argument('--batch_size', type=int, default=36)
args_ps.add_argument('--worker', type=int, default=8, help="number of workers")
args_ps.add_argument('--pre_process', type=int, default=0, help='')
args_ps.add_argument('--init_fc', type=int, default=0, help='')

args_ps.add_argument('--task_list', default='0,1', type=str)

args = args_ps.parse_args()

def run(repeat_time, ex_str, ex_value):
	Setting = ''
	for l in range(len(ex_str)):
		Setting += ' '
		Setting += ex_str[l]
		Setting += ' '
		Setting += str(ex_value[l])

	Default_cmd ='python PWANN.py --dset office_home --net ResNet50 '
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

prepare_data.OfficeHome(root='data/office_home/images/')

for task in task_list:
	st = ST[task]
	run(1, ['--s', '--t', '--max_iterations', '--batch_size', '--worker',
	                         '--point_mass', '--trade_off', '--d_iter', '--d_norm',
	                         '--d_hidden', '--lr_D', '--d_weight_label', '--cls_weight', 
	                         '--opt_G', '--entropy', '--entropy_s', '--lr',
	                        '--pre_process', '--init_fc'],
		[ st[0], st[1], '10000', args.batch_size, args.worker,
		 args.point_mass, args.trade_off, args.d_iter, args.d_norm,
		 args.d_hidden, args.lr_D, args.d_weight_label, args.cls_weight, 
		 args.opt_G, args.entropy, args.entropy_s, args.lr_G,
		  args.pre_process, args.init_fc])