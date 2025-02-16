import os

import imageio

import numpy as np

from dsec_dataset_lite.data.event2frame import EventSlicer
from dsec_dataset_lite.data.event2frame import rectify_events, cumulate_spikes_into_frames

import h5py

from tqdm import tqdm


def generate_files(root: str, sequence: str, num_frames_per_ts: int = 1):

    flow_path = os.path.join(root, 'train', sequence, 'flow', 'forward')

    save_path_flow = os.path.join(root, 'saved_flow_data', 'gt_tensors')
    save_path_mask = os.path.join(root, 'saved_flow_data', 'mask_tensors')

    _create_flow_maps(sequence, flow_path, save_path_flow, save_path_mask)


    timestamps = np.loadtxt(os.path.join(root, 'train', sequence, 'flow', 'forward_timestamps.txt'), delimiter = ',', dtype='int64')

    eventsL_path = os.path.join(root, 'train', sequence, 'events', 'left')


    save_path_events = os.path.join(root, 'saved_flow_data', 'event_tensors',  '{}frames'.format(str(num_frames_per_ts).zfill(2)))
    print(save_path_events)
    #_load_events(sequence, num_frames_per_ts, eventsL_path, timestamps, save_path_events)






def _create_flow_maps(sequence: str, flow_maps_path, save_path_flow, save_path_mask):

    flow_maps_list = os.listdir(flow_maps_path)
    flow_maps_list.sort()

    img_idx = 0

    for flow_map in flow_maps_list:

        img_idx += 1

        path_to_flowfile = os.path.join(flow_maps_path, flow_map)

        flow_16bit = imageio.imread(path_to_flowfile, format='PNG-FI')

        flow_x = (flow_16bit[:,:,0].astype(float) - 2**15) / 128.
        flow_y = (flow_16bit[:,:,1].astype(float) - 2**15) / 128.
        valid_pixels = flow_16bit[:,:,2].astype(bool)

        valid_pixels = valid_pixels[240 - 100:240 + 100, 320 - 100: 320 + 100]

        flow_x = flow_x[240 - 100:240 + 100, 320 - 100: 320 + 100]
        flow_y = flow_y[240 - 100:240 + 100, 320 - 100: 320 + 100]


        flow_x = np.expand_dims(flow_x, axis=0)  # shape (H, W) --> (1, H, W)
        flow_y = np.expand_dims(flow_y, axis=0)


        flow_map = np.concatenate((flow_x, flow_y), axis = 0).astype(np.float32)

        filename = '{}_{}.npy'.format(sequence, str(img_idx).zfill(4))

        np.save(os.path.join(save_path_flow, filename), flow_map)
        np.save(os.path.join(save_path_mask, filename), valid_pixels)


def _load_events(sequence, num_frames_per_ts, events_path, timestamps, save_path_events):

    # load data
    datafile_path = os.path.join(events_path, "events.h5")
    datafile = h5py.File(datafile_path, 'r')
    event_slicer = EventSlicer(datafile)

    N_chunks = timestamps.shape[0]  # N_chunks = N_grountruths

    fileidx = 0

    for numchunk in tqdm(range(N_chunks)):

        fileidx += 1

        t_beg, t_end = timestamps[numchunk]
        dt = (t_end - t_beg) / num_frames_per_ts

        chunk = []

        for numframe in range(num_frames_per_ts):

            t_start = t_beg + numframe * dt
            t_end = t_beg + (numframe + 1) * dt

            # load events within time window
            event_data = event_slicer.get_events(t_start, t_end)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            # rectify events
            rectmap_path = os.path.join(events_path, "rectify_map.h5")
            rectmap_file = h5py.File(rectmap_path)
            rectmap = rectmap_file['rectify_map'][()]

            xy_rect = rectify_events(x, y, rectmap)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            # cumulate events
            frame = cumulate_spikes_into_frames(x_rect, y_rect, p)
            chunk.append(frame)

        # format into chunks
        chunk = np.array(chunk).astype(np.float32)

        filename = '{}_{}.npy'.format(sequence, str(fileidx).zfill(4))


        np.save(os.path.join(save_path_events, filename), chunk)


    # close hdf5 files
    datafile.close()
    rectmap_file.close()

root = "E:\\DSEC\\"
sequences = os.listdir(os.path.join(root, 'train'))
num_to_load = len(sequences)
for i in range(num_to_load):
    if sequences[i] != "saved_flow_data" and sequences[i] != "flow_2_event_idx_mapping" and sequences[i] != "env_names.txt" and os.path.isdir(os.path.join(root, 'train', sequences[i], "flow")):
        print("Sequence: ", sequences[i])
        generate_files(root, sequences[i], 11)