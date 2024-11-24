import subprocess

import get_args

args_ps = get_args.get()
args = args_ps.parse_args()

########## Only used in OfficeHome and DomainNet
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
seed_list = [int(i) for i in args.seed_list.strip().split(',')]

for seed in seed_list:
	for task in task_list:
		st = ST[task]
		args_string = ' '.join([f'--{k} {v}' for k, v in vars(args).items()])
		extra_string = f' --seed {seed} --s {st[0]} --t {st[1]}'
		full_string = args_string + extra_string
		cmd = 'python PWANN.py ' + full_string
		print(f'Running : {cmd} \n')
		subprocess.run(cmd, shell=True)