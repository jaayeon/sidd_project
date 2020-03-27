#!/usr/bin/env python
# from skimage.measure import compare_ssim
# from skimage.measure import compare_psnr

import sys, os
import glob
import time
import numpy as np
import os.path
import shutil
import math

from scipy.io.matlab.mio import savemat, loadmat

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt

from options import args
import data.make_patches as mp
from data.make_patches import pad_img, unpad_img
from utils.saver import load_model, load_config

from models import set_model
from utils.helper import set_gpu, set_test_dir
from utils.wavelet import swt2d, iswt2d, iswt2d_rgb, unnormalize_coeffs

from data.images import ImageDataset, SWTImageDataset


def augment(img, flip=0, rot90=0):
    if img.ndim==2:
        if flip : img = img[:, ::-1]
        if rot90 : img = np.rot90(img, rot90)
    elif img.ndim==3 :
        if flip : img = img[:, ::-1, :]
        if rot90 : img = np.rot90(img, rot90,(0,1))
    return img

def de_augment(img, rot90=0, flip=0):
    if img.ndim==2:
        if rot90 : img = np.rot90(img, rot90)
        if flip : img = img[:, ::-1]
    elif img.ndim==3 :
        if rot90 : img = np.rot90(img, rot90,(0,1))
        if flip : img = img[:, ::-1, :]
    return img    


def denoiser_inside(opt, net, input_img):

    start_time = time.time()

    if opt.model == 'waveletdl':
        img_patch_dataset = SWTImageDataset(opt, input_img)
        img_patch_dataloader = DataLoader(dataset=img_patch_dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=False)
        approx_list = img_patch_dataset.get_approx_list()
    else:
        img_patch_dataset = ImageDataset(opt, input_img)
        img_patch_dataloader = DataLoader(dataset=img_patch_dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=False)
    
    img_shape = img_patch_dataset.get_img_shape()
    pad_img_shape = img_patch_dataset.get_padded_img_shape()

    out_list = []

    for idx, batch in enumerate(img_patch_dataloader, 1):
        with torch.no_grad():
            x = batch
            
            if opt.use_cuda:
                x = x.to(opt.device)
            
            out = net(x)
            if opt.use_cuda:
                out = out.to('cpu').detach().numpy()
            else:
                out = out.detach().numpy()

            out_list.append(out)

    out = np.concatenate(out_list, axis=0)
    
    if opt.model == 'waveletdl' :
        out_list = np.zeros((out.shape[0], out.shape[2], out.shape[3], opt.n_channels), dtype=input_img.dtype)
        out_list = out_list.squeeze()

        for i in range(out.shape[0]):

            if opt.n_channels == 1:
                inv_swt = iswt2d(approx_list[i], out[i], wavelet=opt.wavelet_func)
                inv_swt[inv_swt > 1.0] = 1.0
                inv_swt[inv_swt < 0] = 0
                out_list[i] = inv_swt
            else:
                inv_swt = iswt2d_rgb(approx_list[i], out[i], wavelet=opt.wavelet_func)
                # print("inv_swt.max():", np.amax(inv_swt))
                # print("inv_swt.min():", np.amin(inv_swt))
                inv_swt[inv_swt > 1.0] = 1.0
                inv_swt[inv_swt < 0] = 0
                inv_swt *= 255
                inv_swt = inv_swt.astype(np.uint8)
                out_list[i] = inv_swt
                
        out = out_list
        out = out.squeeze()
    else :
        out[out>1.0] = 1.0
        out[out<0] = 0
        out = out.squeeze()
        if opt.n_channels == 3 :
            out *= 255
            out = out.astype(np.uint8)
            out = np.ascontiguousarray(out.transpose((0,2,3,1)))

    # print(out.shape)
    if opt.test_patches:
        out_img = mp.recon_patches(out, pad_img_shape[1], pad_img_shape[0], opt.patch_size, opt.patch_offset)
        out_img = unpad_img(out_img, opt.patch_offset, img_shape)
    else:
        out_img = out

    # print("===>[*] Total Time: {:4f}s\n".format(time.time() - start_time))

    return out_img


def denoiser(opt, net, input_img):  

    DN = denoiser_inside

    input_shape = input_img.shape
    if opt.ensemble : 
        out_img = np.zeros(input_shape, dtype = np.float32)
        for f in range(2) :
            for r in range(4):
                print('ensemble')
                aug_input = augment(input_img, flip=f, rot90=r)
                aug_out = DN(opt, net, aug_input)
                out_img += de_augment(aug_out, rot90=4-r, flip=f)  
        out_img = out_img/8.0
        out_img = out_img.astype(np.float32)
    else : 
        out_img = DN(opt, net, input_img)

    return out_img
    

def prep_Result(opt):

    net = set_model(opt)
    _, net, _ = load_model(opt, net)

    if opt.n_channels == 1:
        from skimage.external.tifffile import imsave, imread
    else:
        from skimage.io import imsave, imread

    opt = set_gpu(opt)

    if opt.use_cuda:
        net = net.to(opt.device)

    if opt.multi_gpu:
        net = nn.DataParallel(net)

    set_test_dir(opt)
    if not os.path.exists(opt.test_result_dir):
        os.makedirs(opt.test_result_dir)

    res_img_dir = os.path.join(opt.test_result_dir, 'result_img_dir')
    if not os.path.exists(res_img_dir):
        os.makedirs(res_img_dir)
    
    # create results directory
    res_dir = os.path.join(opt.test_result_dir, 'res_dir')
    os.makedirs(os.path.join(res_dir), exist_ok=True)

    print('\ntest_result_dir : ', opt.test_result_dir)
    print('\nresult_img_dir : ', res_img_dir)
    print('\nresult_dir : ', res_dir)

    # total_psnr = 0
    # total_ssim = 0

    loss_criterion = nn.MSELoss()
    total_psnr = 0.0

    if opt.n_channels == 1:
        # load noisy images
        noisy_fn = 'siddplus_test_noisy_raw.mat'
        noisy_key = 'siddplus_test_noisy_raw'
        noisy_mat = loadmat(os.path.join(opt.test_dir, opt.dataset,  noisy_fn))[noisy_key]

        # denoise
        n_im, h, w = noisy_mat.shape
        results = noisy_mat.copy()

        start_time = time.time()
        for i in range(n_im):
            print('\n[*]PROCESSING..{}/{}'.format(i,n_im))

            noisy = np.reshape(noisy_mat[i, :, :], (h, w))
            denoised = denoiser(opt, net, noisy)
            results[i, :, :] = denoised

            result_name = str(i) + '.tiff'
            concat_img = np.concatenate((denoised, noisy), axis = 1)
            imsave(os.path.join(res_img_dir, result_name), concat_img )

            denoised = torch.Tensor(denoised)
            noisy = torch.Tensor(noisy)

            mse_loss = loss_criterion(denoised, noisy)
            psnr = 10 * math.log10(1 / mse_loss.item())
            total_psnr += psnr
            print('%.5fs .. [%d/%d] psnr : %.5f, avg_psnr : %.5f'%(time.time()-start_time, i, n_im, psnr, total_psnr/(i+1)))

    else :
        # load noisy images
        noisy_fn = 'siddplus_test_noisy_srgb.mat'
        noisy_key = 'siddplus_test_noisy_srgb'
        noisy_mat = loadmat(os.path.join(opt.test_dir, opt.dataset, noisy_fn))[noisy_key]

        # denoise
        n_im, h, w, c = noisy_mat.shape
        results = noisy_mat.copy()

        start_time = time.time()
        for i in range(n_im):
            print('\n[*]PROCESSING..{}/{}'.format(i,n_im))

            noisy = np.reshape(noisy_mat[i, :, :, :], (h, w, c))
            denoised = denoiser(opt, net, noisy)
            results[i, :, :, :] = denoised

            result_name = str(i) + '.png'
            concat_img = np.concatenate((denoised, noisy), axis = 1)
            imsave(os.path.join(res_img_dir, result_name), concat_img )

            denoised = torch.Tensor(denoised).float()/255.0
            noisy = torch.Tensor(noisy).float()/255.0

            mse_loss = loss_criterion(noisy, denoised)
            psnr = 10 * math.log10(1 / mse_loss.item())
            total_psnr += psnr
            print('%.5fs .. [%d/%d] psnr : %.5f, avg_psnr : %.5f'%(time.time()-start_time, i, n_im, psnr, total_psnr/(i+1)))


    print("****total avg psnr : %.10f", total_psnr/(n_im))
    # save denoised images in a .mat file with dictionary key "results"
    res_fn = os.path.join(res_dir, 'results.mat')
    res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
    savemat(res_fn, {res_key: results})

    runtime = 0.0  # seconds / megapixel
    cpu_or_gpu = 0  # 0: GPU, 1: CPU
    use_metadata = 0  # 0: no use of metadata, 1: metadata used
    other = '(optional) any additional description or information'

    # prepare and save readme file
    readme_fn = os.path.join(res_dir, 'readme.txt')  # Note: do not change 'readme.txt'
    with open(readme_fn, 'w') as readme_file:
        readme_file.write('Runtime (seconds / megapixel): %s\n' % str(runtime))
        readme_file.write('CPU[1] / GPU[0]: %s\n' % str(cpu_or_gpu))
        readme_file.write('Metadata[1] / No Metadata[0]: %s\n' % str(use_metadata))
        readme_file.write('Other description: %s\n' % str(other))

    # compress results directory
    res_zip_fn = 'results_dir'
    shutil.make_archive(os.path.join(opt.test_result_dir, res_zip_fn), 'zip', res_dir)
