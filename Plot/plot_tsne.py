import matplotlib.pyplot as plt
import argparse
import pdb
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import current_process, Pool




norm = plt.Normalize(vmin=0., vmax=1.)



ags_ps = argparse.ArgumentParser()
ags_ps.add_argument('folder', type=str)
ags_ps.add_argument('--tags', type=str, default='Test/t_acc')
ags_ps.add_argument('--show', type=str, default='0')
ags_ps.add_argument('--label_color', type=str, default='0')


args = ags_ps.parse_args()


def main(folder, color_label=True):
	All_task = sorted(os.listdir(folder))


	for i in All_task:
		task_name = os.path.join(folder, i)
		# print(i)
		print(task_name)
		if not os.path.isdir(task_name):
			continue

		files = os.listdir(task_name)
		tsne_name = None
		for j in files:
			if 'TSNE.png' in j:
				tsne_name = j
				break
		if tsne_name is None:
			raise ValueError('No tsne name is founded.')
		else:
			print('old tsne image: {}'.format(tsne_name))
		step_name = tsne_name.split('_')[0]

		tsne_file = os.path.join(task_name, step_name + '_tsne.npy')
		discard_index_source_file = os.path.join(task_name, step_name + '_discard_index_source.npy')
		keep_index_source_file = os.path.join(task_name, step_name + '_keep_index_source.npy')
		index_target_file = os.path.join(task_name, step_name + '_index_target.npy')
		source_label_cut_file = os.path.join(task_name, step_name + '_source_label_cut.npy')
		target_label_file = os.path.join(task_name, step_name + '_target_label.npy')

		print('Loading....')
		# print(tsne_file, discard_index_source_file, keep_index_source_file, index_target_file, source_label_cut_file, target_label_file)

		X_tsne = np.load(tsne_file)
		discard_index_source = np.load(discard_index_source_file)
		keep_index_source = np.load(keep_index_source_file)
		index_target = np.load(index_target_file)
		source_label_cut = np.load(source_label_cut_file)
		target_label = np.load(target_label_file)

		plt.figure(figsize=(5, 5))

		print('Plotting...')
		if color_label == True:
			plt.scatter(X_tsne[discard_index_source, 0], X_tsne[discard_index_source, 1], c='gray', s=10, alpha=0.8,
			            marker='+')
			plt.scatter(X_tsne[keep_index_source, 0], X_tsne[keep_index_source, 1], c=source_label_cut / 10 + 0.05,
			            cmap=plt.cm.tab20, norm=norm, s=10, alpha=0.8, marker='+')
			plt.scatter(X_tsne[index_target, 0], X_tsne[index_target, 1], c=target_label / 10, cmap=plt.cm.tab20,
			            norm=norm, s=10, alpha=0.8, marker='o')
		else:
			plt.scatter(X_tsne[discard_index_source, 0], X_tsne[discard_index_source, 1], color=(0.7, 0.7, 0.7), s=10, alpha=0.8,
			            marker='+')
			plt.scatter(X_tsne[keep_index_source, 0], X_tsne[keep_index_source, 1], color='r', s=10, alpha=0.3,
			            marker='+')
			plt.scatter(X_tsne[index_target, 0], X_tsne[index_target, 1], color='b', s=10, alpha=0.3, marker='o')

		print('Saving to {}'.format(os.path.join(task_name, 'new_' + 't.png')))
		plt.tight_layout()
		plt.savefig(os.path.join(task_name, 'new_' + 't.png'))

if __name__ == '__main__':
    folder_list = args.folder.strip().split(',')
    print(folder_list)
    p = Pool(8)


    for f in folder_list:
        p.apply_async(main, [f, args.tags, args.show, args.label_color])
        # p.apply(main, [f, args.label_color])
        # main(f, args.tags, args.show)

    p.close()
    p.join()
