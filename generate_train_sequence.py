import pandas as pd
import os
import numpy as np

root = 'E:\\EBAL_v10\\train\\saved_flow_data\\'
num_frames_per_ts = "11"
events_path = os.path.join(root, 'event_tensors', '{}frames'.format(str(num_frames_per_ts).zfill(2)))

files = np.array(os.listdir(events_path))

df_files = pd.Series(files)
#df_files = df_files[~df_files.str.contains('N12')]
ventral_files = np.array(df_files[ df_files.str.contains('V') ])
nominal_files = np.array(df_files[ df_files.str.contains('N') ])
divert_files = np.array(df_files[ df_files.str.contains('D') ])

mode_list = [ventral_files, nominal_files, divert_files]

train_list = []
valid_list = []

for mode in mode_list:

    num_files = len(mode)

    end_coefficient = 1
    split_coefficient = end_coefficient * 0.8
    

    i_split = int(split_coefficient * len(mode))
    i_end = int(end_coefficient * len(mode))

    file_pairs_train = [(mode[i], mode[i+1]) for i in range (0, i_split - 1)]

    df_train = pd.DataFrame(file_pairs_train, columns = ['f1', 'f2'])

    file_pairs_valid = [(mode[i], mode[i+1]) for i in range (i_split, i_end - 1)]

    df_valid = pd.DataFrame(file_pairs_valid, columns = ['f1', 'f2'])

    train_list.append(df_train)
    valid_list.append(df_valid)

df_train_all = pd.concat(train_list, axis=0)
df_valid_all = pd.concat(valid_list, axis=0)

df_train_all.to_csv('train_split_landing.csv', header = None, index = None)
df_valid_all.to_csv('valid_split_landing.csv', header = None, index = None)