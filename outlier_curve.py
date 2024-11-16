import os
import argparse
import numpy as np

agsps = argparse.ArgumentParser()
agsps.add_argument('name', type=str, default='G:\Mypaper\PWAN\Process\Figure\Illustation\curve\AC')
args = agsps.parse_args()

All = sorted(os.listdir(args.name))

Inlier_ratio = []
Accuracy = []

for i in All:
	if i.endswith('_target_label_predict.npy'):
		index = i[0:-25]
		# target_label_predict_filename = '{}_target_label_predict.npy'.format(args.name)
		target_label_predict = np.load(os.path.join(args.name, i))
		All_predict = np.argmax(target_label_predict, 1)
		Inlier_ratio.append(np.sum(All_predict < 25) / All_predict.shape[0])

		T_name = index + '_target_label.npy'
		target_label_True = np.load(os.path.join(args.name, T_name))

		Accuracy.append(np.sum(target_label_True == All_predict) / All_predict.shape[0])

print(Inlier_ratio)
print(Accuracy)