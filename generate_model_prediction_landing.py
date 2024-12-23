import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as TvT

from torch.autograd import Variable

from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import neuron

from network_3d.poolingNet_cat_1res_200 import NeuronPool_Separable_Pool3d_200


from tqdm import tqdm

from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite

import numpy as np

import math
import os


flight_list = ["N9", "N10", "N11", "N12", "D9", "D10", "D11", "D12", "V9", "V10", "V11", "V12"]

for flight in flight_list:
    root_folder = f'/root/saved_flow_data_test/'
    sequence_list = f'/root/saved_flow_data_test/sequence_lists/{flight}.csv'

    if not os.path.isdir(f'/root/results/tests/'):
        os.mkdir(f'/root/results/tests/')

    checkpoint = f'/root/results/checkpoints_and_logs/checkpoint_epoch31.pth'
    results_directory = f'/root/results/{flight}/'

    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    results_directory = os.path.join(results_directory, flight)
    # Enable GPU
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    ################################
    ## DATASET LOADING/GENERATION ##
    ################################

    num_frames_per_ts = 1
    forward_labels = 1

    # Create validation dataset
    print("Creating Validation Dataset ...")
    valid_dataset = DSECDatasetLite(root = root_folder, file_list = sequence_list, num_frames_per_ts = 11, transform = None)

    # Define validation dataloader
    valid_dataloader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False, drop_last = False, pin_memory = True)

    ########################
    ## TRAINING FRAMEWORK ##
    ########################

    # Create the network

    net = NeuronPool_Separable_Pool3d_200().to(device)
    net.load_state_dict(torch.load(checkpoint))


    ##########
    ## TEST ##
    ##########

    # Validation Datasetimport random

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import torchvision.transforms as TvT

    from torch.autograd import Variable

    from spikingjelly.clock_driven import functional
    from spikingjelly.clock_driven import neuron

    from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d


    from tqdm import tqdm

    from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite

    import numpy as np

    from eval.progress_plot_full_v2 import plot_evolution

    import math
    import os


    pred_sequence = []
    label_sequence = []
    mask_sequence = []

    net.eval()
    print('Validating... (test sequence)')

    net.eval()
    for chunk, mask, label in tqdm(valid_dataloader):

        functional.reset_net(net)
        chunk = torch.transpose(chunk, 1, 2)


        mask = torch.unsqueeze(mask, dim = 1)
        mask = torch.cat((mask, mask), axis = 1)

        label = label.to(device = device, dtype = torch.float32) # [num_batches, 2, H, W]

        mask = mask.to(device = device)

        with torch.no_grad():
            _, _, _, pred = net(chunk)

        pred_sequence.append(torch.squeeze(pred[0,:,:,:]).cpu().detach().numpy())
        label_sequence.append(torch.squeeze(label[0,:,:,:]).cpu().detach().numpy())
        mask_sequence.append(torch.squeeze(mask[0]).cpu().detach().numpy())

    # Video generation
    pred_sequence = np.array(pred_sequence)
    label_sequence = np.array(label_sequence)


    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    np.save(f'{results_directory}/pred_sequence.npy', pred_sequence)
    np.save(f'{results_directory}/label_sequence.npy', label_sequence)
    np.save(f'{results_directory}/mask_sequence.npy', mask_sequence)

    print('SO FAR, EVERYTHING IS WORKING!!!')
