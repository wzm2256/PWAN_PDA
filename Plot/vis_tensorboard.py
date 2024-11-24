import pdb
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import current_process, Pool

ags_ps = argparse.ArgumentParser()
ags_ps.add_argument('folder', type=str)
ags_ps.add_argument('--tags', type=str, default='Test/t_acc')
ags_ps.add_argument('--show', type=str, default='0')
ags_ps.add_argument('--value', type=str, default='value')
ags_ps.add_argument('--silence', type=int, default=0)

args = ags_ps.parse_args()


def main(folder, tags, show, silence, value):
    All_task = sorted(os.listdir(folder))

    print('----' + folder.split('/')[-2] + '----')
    print('Task number: {}'.format(len(   [ i for i in All_task if os.path.isdir(os.path.join(folder, i))])))

    All_task_name = []
    All_record = []

    Error_flag = 0
    for i in All_task:
        # pdb.set_trace()
        task_name = os.path.join(folder, i)

        if not os.path.isdir(task_name):
            continue
        log_dir = os.path.join(task_name, '0', 'LOG')

        try:
            log_file = os.listdir(log_dir)
        except:
            return -1

        if len(log_file) > 1:
            print('Error: multiple {}'.format(log_dir))

        event_path = os.path.join(log_dir, log_file[0])
        ea = event_accumulator.EventAccumulator(event_path,
                                                size_guidance={  # see below regarding this argument
                                                    event_accumulator.COMPRESSED_HISTOGRAMS: 1,
                                                    event_accumulator.IMAGES: 1,
                                                    event_accumulator.AUDIO: 1,
                                                    event_accumulator.SCALARS: 0,
                                                    event_accumulator.HISTOGRAMS: 1,
                                                })
        ea.Reload()  # loads events from fileea.Scalars('Max_grad')

        if value == 'wall_time':
            try:
                time_list = np.array(pd.DataFrame(ea.Scalars(tags))[value])
                Total_time = time_list[-1] - time_list[0]
                print('Total_time: {}'.format(Total_time))
            except:
                Total_time = -1
                print('Total_time: {}'.format(Total_time))
            return Total_time

        try:
            curve = np.array(pd.DataFrame(ea.Scalars(tags))['value'])
        except:
            Error_flag = 1
            print('Error loading: {}'.format(i))
            continue

        All_task_name.append(task_name)
        All_record.append(curve)
        curve_file = os.path.join(task_name, 'curve.txt')
        f_curve = open(curve_file, 'w')
        for acc in curve:
            f_curve.write(str(np.round(acc, 5)) + '\n')
        f_curve.close()

    if Error_flag == 1:
        return

    ### plot all
    for i in range(len(All_task_name)):
        plt.plot(All_record[i], label=All_task_name[i].split('/')[-1])
    plt.legend()
    
    if show == '1':
        plt.show()
    else:
        plt.savefig(os.path.join(folder, 'all.png'))
    
    plt.close()
    
    ## summary all
    save_folder = os.path.join(folder, 'best.txt')
    f = open(save_folder, 'w')
    
    Best = []
    for i in range(len(All_task_name)):
        best = np.max(All_record[i])
    
        f.write('{}\t{}\n'.format(All_task_name[i].split('/')[-1], best))
        Best.append(best)
    
    f.write('\n')
    f.write('{}\t{}\n'.format('Averaged', np.mean(np.array(Best))))
    
    f.close()
    
    ## plot average
    Max_len = 10000
    All_len = []
    for i in range(len(All_record)):
        L = len(All_record[i])
        All_len.append(L)
        if silence == 0:
            print(L)
        # pdb.set_trace()
        All_record[i] = np.pad(All_record[i], (0, 10000 - len(All_record[i])), 'constant', constant_values=np.nan)
        # tmp_list = list(All_record[i])
        # tmp_list.append([np.nan] * (100 - len(All_record[i])))
        # np.array(tmp_list)
    
    # pdb.set_trace()
    Max_len = np.max(np.array(All_len))
    avg_acc = np.nanmean(All_record, 0)
    Clip_epoch =  np.nanargmax(avg_acc)
    #

    print('Maximum precision in epoch {}: \t{}'.format(Clip_epoch, avg_acc[Clip_epoch]))
    print('Final precision in epoch {}: \t{}'.format(Max_len, avg_acc[Max_len-1]))
    if silence == 0:
        print(avg_acc[:Max_len])
    plt.plot(avg_acc[:Max_len])

    avg_curve = open(os.path.join(folder, 'avg_curve.txt'), 'w')
    for i in range(Max_len):
        avg_curve.write(str(np.round(avg_acc[i], 4)))
        avg_curve.write('\t')
    avg_curve.close()

    if show == '1':
        plt.show()
    else:
        plt.savefig(os.path.join(folder, 'avg.png'))

    plt.close()
    ## summary latest based on average
    
    save_folder = os.path.join(folder, 'latest.txt')
    f = open(save_folder, 'w')
    
    Record = []
    for i in range(len(All_task_name)):
        tmp = All_record[i][Clip_epoch]
    
        f.write('{}\t{}\n'.format(All_task_name[i].split('/')[-1], tmp))
        Record.append(tmp)
    
    f.write('\n')
    f.write('{}\t{} at {} epoch.\n'.format('Averaged', np.mean(np.array(Record)), Clip_epoch))
    
    f.write('{}\t{} at {} epoch.\n'.format('Averaged', avg_acc[Max_len-1], Max_len-1))
    
    f.close()


if __name__ == '__main__':
    folder_list = args.folder.strip().split(',')
    # print(folder_list)
    # p = Pool(8)


    for f in folder_list:
        # p.apply_async(main, [f, args.tags, args.show, args.silence])
        main(f, args.tags, args.show, args.silence, args.value)

    # p.close()
    # p.join()
