import os

import imageio

import numpy as np

#from .event2frame import EventSlicer
#from .event2frame import rectify_events, cumulate_spikes_into_frames

import h5py

from tqdm import tqdm

import math
from numba import jit

import hdf5plugin
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
from typing import Dict, Tuple

# CODE BORROWED AND ADAPTED FROM THE DSEC GIT REPOSITORY
# (https://github.com/uzh-rpg/DSEC)


def rectify_events(x: np.ndarray, y: np.ndarray, rectify_map):


    height = 180; width = 240

    assert rectify_map.shape == (height, width, 2), rectify_map.shape
    assert x.max() < width
    assert y.max() < height
    return rectify_map[y, x]


def cumulate_spikes_into_frames(X_list, Y_list, P_list):

    frame = np.zeros((2, 200, 200), dtype='float')

    for x, y, p in zip(X_list, Y_list, P_list):
        if p == 1:
            frame[0, y, x] += 1  # register ON event on channel 0
        else:
            frame[1, y, x] += 1  # register OFF event on channel 1

    return frame


class EventSlicer:
    def __init__(self, h5f: h5py.File):

        self.h5f = h5f

        self.events = dict()

        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9

        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        #  0: erase initial zeroes
        #self.ms_to_idx_orig = self.ms_to_idx
        #import sys
        #np.set_printoptions(threshold=sys.maxsize)
        #print(len(self.ms_to_idx))

        #self.ms_to_idx = np.append([0], self.ms_to_idx[self.ms_to_idx != 0])

        #print(self.ms_to_idx)
        #print(self.ms_to_idx.shape)
        #print(self.events['t'].shape)

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

        # 0: set offset to compensate for the initial zeroes
        #print(int((self.ms_to_idx_orig.shape[0] - self.ms_to_idx.shape[0])))
        #self.t_offset = -int((self.ms_to_idx_orig.shape[0] - self.ms_to_idx.shape[0]) * 1000)

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        #t_start_us -= self.t_offset
        #t_end_us -= self.t_offset

        t_start_us -= self.t_offset 
        t_end_us -= self.t_offset 

        #print("t_start_us", t_start_us)
        #print("t_end_us", t_end_us)

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)

        t_start_ms_idx = self.ms2idx(t_start_ms)
        #print(t_start_ms_idx)
        t_end_ms_idx = self.ms2idx(t_end_ms)
        #print(t_end_ms_idx)

        #print("t_start_ms", t_start_ms)
        #print("t_end_ms", t_end_ms)
        #print("t_start_ms_idx", t_start_ms_idx)
        #print("t_end_ms_idx", t_end_ms_idx)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        if len((self.events['t'][t_start_ms_idx:t_end_ms_idx])) > 0:
            time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
            #print("time_array_conservative", time_array_conservative)
            idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
            #print("idx_start_offset",idx_start_offset)
            #print("idx_end_offset",idx_end_offset)
            t_start_us_idx = t_start_ms_idx + idx_start_offset
            t_end_us_idx = t_start_ms_idx + idx_end_offset
            #print("t_start_us_idx", t_start_us_idx)
            #print("t_end_us_idx", t_end_us_idx)
            # Again add t_offset to get gps time
            events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
            #print("t", events['t'])
            if len(events['t']) == 0:
                print("len events 0 in chunk")
            for dset_str in ['p', 'x', 'y']:
                events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
                assert events[dset_str].size == events['t'].size
            #print(events) 
            return events
        else: 
            return None


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        #print("time_array", time_array)
        #print("time_start_us", time_start_us)
        #print("time_end_us", time_end_us)

        assert time_array.ndim == 1
        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000
            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    #print('time array', time_array[idx_from_start])
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


def generate_files(root: str, sequence: str, num_frames_per_ts: int = 1):

    flow_path = os.path.join(root, sequence, 'flow', 'forward')

    save_path_flow = os.path.join(root, 'saved_flow_data', 'gt_tensors')
    save_path_mask = os.path.join(root, 'saved_flow_data', 'mask_tensors')

    _create_flow_maps(sequence, flow_path, save_path_flow, save_path_mask)


    timestamps = np.loadtxt(os.path.join(root, sequence, 'flow', 'forward_timestamps.txt'), delimiter = ',', dtype='int64')

    events_path = os.path.join(root, sequence)


    save_path_events = os.path.join(root, 'saved_flow_data', 'event_tensors',  '{}frames'.format(str(num_frames_per_ts).zfill(2)))
    print(save_path_events)
    _load_events(sequence, num_frames_per_ts, events_path, timestamps, save_path_events)


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
            try:
                p = event_data['p']
                t = event_data['t']
                x = event_data['x']
                y = event_data['y']
            except:
                p = []
                t = []
                x = []
                y = []

            # cumulate events
            frame = cumulate_spikes_into_frames(x, y, p)
            chunk.append(frame)

        # format into chunks
        chunk = np.array(chunk).astype(np.float32)

        filename = '{}_{}.npy'.format(sequence, str(fileidx).zfill(4))


        np.save(os.path.join(save_path_events, filename), chunk)


    # close hdf5 files
    datafile.close()

root = "/media/odvorak/Expansion/ALED_v30/test"
sequences = os.listdir(root)
num_to_load = len(sequences)
for i in range(num_to_load):
    if sequences[i] != "saved_flow_data" and sequences[i] != "flow_2_event_idx_mapping" and sequences[i] != "env_names.txt":
        print("Sequence: ", sequences[i])
        generate_files(root, sequences[i], 11)

