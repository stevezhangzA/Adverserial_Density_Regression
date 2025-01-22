import numpy as np
import pickle as pkl
import os

def print_and_build_table(root_path):
    loggings = {}
    for task_name in os.listdir(root_path):
        sub_foler = os.path.join(root_path, task_name)
        exp_record = []
        for sub_dir in os.listdir(sub_foler):
            if sub_dir.split('.')[-1] == 'pkl':
                filename = os.path.join(sub_foler, sub_dir)
                try:
                    readout_fp = pkl.load(open(filename, 'rb'))
                except:
                    continue
                # if len(readout_fp)==1:
                try:
                    if isinstance(readout_fp[0], list):
                        max_data = max(readout_fp[0])
                        exp_record.append(max_data)
                except:
                    continue
            else:
                continue
        exp_record = np.array(exp_record)
        mean, std = np.around(np.mean(exp_record), 1), np.around(np.std(exp_record), 1)
        print(exp_record)
        print(task_name, str(mean) + '$\pm$' + str(std))
        loggings[task_name] = exp_record
        print('\n')
    return loggings

if __name__ == '__main__':
    lfd_sets = ['lfd_1_vq']
    lfd_records_vq = []
    for lfd_subdir in lfd_sets:
        loaded_data = print_and_build_table(lfd_subdir)
        lfd_records_vq.append(loaded_data)

