import torch
import numpy as np

def mod_loss_function(pred, label, mask):

    n_pixels = torch.sum(mask)
    #print(torch.sum(pred), torch.sum(label))    
    #print(pred)
    error_mod = torch.sqrt(torch.pow(pred[:,0] - label[:,0], 2) + torch.pow(pred[:,1] - label[:,1], 2))
    
    return torch.sum(error_mod * mask) / n_pixels

def thesis_loss_function(pred, label, mask):
    n_pixels = torch.sum(mask)
    # print(torch.sum(pred), torch.sum(label))
    # print(pred)
    error_mod = torch.sqrt(torch.pow(pred[:, 0] - label[:, 0], 2) + torch.pow(pred[:, 1] - label[:, 1], 2))
    sig_u = pred[:, 0] - label[:, 0]
    sig_v = pred[:, 1] - label[:, 1]
    # unsig_u_down = torch.nn.functional.avg_pool2d(unsig_u, kernel_size=50, stride=50)
    # unsig_v_down = torch.nn.functional.avg_pool2d(unsig_v, kernel_size=50, stride=50)
    # corr_loss = torch.abs(unsig_u_down) + torch.abs(unsig_v_down)
    # return torch.sum(error_mod * mask) / n_pixels + 0.5 * torch.sum(corr_loss) / 16
    #return torch.sum(error_mod * mask) / n_pixels

    kernel_size = 50

    # Create a mean kernel (normalized)
    mean_kernel = torch.ones((1, 1, kernel_size, kernel_size), device='cuda') / (kernel_size ** 2)

    # Apply the mean filter to smooth the entire field
    unsig_u_smooth = torch.nn.functional.conv2d(sig_u, mean_kernel, padding=kernel_size // 2)
    unsig_v_smooth = torch.nn.functional.conv2d(sig_v, mean_kernel, padding=kernel_size // 2)

    # Compute correlation loss
    corr_loss = torch.abs(unsig_u_smooth) + torch.abs(unsig_v_smooth)

    return torch.sum(error_mod * mask) / n_pixels + 0.5 * torch.sum(corr_loss * mask) / n_pixels

    
def rel_loss_function(pred, label, mask, epsilon = 1e-7):

    n_pixels = torch.sum(mask)
    #print(torch.sum(pred), torch.sum(label))    
    #print(pred)
    error_mod = torch.sqrt(torch.pow(pred[:,0] - label[:,0], 2) + torch.pow(pred[:,1] - label[:,1], 2))
    gt_mod = torch.sqrt(torch.pow(label[:,0], 2) + torch.pow(label[:,1], 2))
    
    return (1/n_pixels) * torch.sum((error_mod * mask) / (gt_mod + epsilon))


def cosine_loss_function(pred, label, mask, epsilon = 1e-7):

    n_pixels = torch.sum(mask)
   
    pred_mod = torch.sqrt(torch.pow(pred[:,0], 2) + torch.pow(pred[:,1], 2))
    label_mod = torch.sqrt(torch.pow(label[:,0], 2) + torch.pow(label[:,1], 2))

    dot_product = pred[:,0]*label[:,0] + pred[:,1]*label[:,1]

    cosine = (dot_product + epsilon) / (pred_mod*label_mod + epsilon)
    #cosine = (dot_product) / (pred_mod*label_mod)
    
    cosine = torch.clamp(cosine, min = -1. + epsilon, max = 1. - epsilon)

    return torch.sum((1. - cosine) * mask) /  n_pixels


def angular_loss_function(pred, label, mask, epsilon = 1e-7):

    n_pixels = torch.sum(mask)
   
    pred_mod = torch.sqrt(torch.pow(pred[:,0], 2) + torch.pow(pred[:,1], 2))
    label_mod = torch.sqrt(torch.pow(label[:,0], 2) + torch.pow(label[:,1], 2))

    dot_product = pred[:,0]*label[:,0] + pred[:,1]*label[:,1]

    cosine = (dot_product + epsilon) / (pred_mod*label_mod + epsilon)
    #cosine = (dot_product) / (pred_mod*label_mod)
    
    cosine = torch.clamp(cosine, min = -1. + epsilon, max = 1. - epsilon)

    return torch.sum(torch.acos(cosine) * mask) /  n_pixels
