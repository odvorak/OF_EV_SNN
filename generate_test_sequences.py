import pandas as pd
import os
import numpy as np


def make_sequence_list(sequence, root):
    num_frames_per_ts = 11
    events_path = os.path.join(root, 'event_tensors', '{}frames'.format(str(num_frames_per_ts).zfill(2)))

    files = np.array(os.listdir(events_path))

    df_files = pd.Series(files)

    files = np.array(df_files[ df_files.str.contains(sequence)])

    num_files = len(files)

    file_pairs = [(files[i], files[i+1]) for i in range (0, num_files - 1)]

    df = pd.DataFrame(file_pairs, columns = ['f1', 'f2'])

    print(root)

    df.to_csv(os.path.join("/media/odvorak/Expansion/ALED_v30/test/saved_flow_data/", "sequence_lists", f'{sequence}.csv'), header = None, index = None)

    return f'{sequence}.csv'

if __name__ == "__main__":
    tests = ["N9", "N10", "N11", "N12", "D9", "D10", "D11", "D12", "V9", "V10", "V11", "V12"]
    for test in tests:
        make_sequence_list(test, '/media/odvorak/Expansion/ALED_v30/test/saved_flow_data/')
