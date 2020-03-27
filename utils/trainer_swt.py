import os
import time
import math
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils.saver import load_model, save_checkpoint, save_config
from utils.helper import set_gpu, set_checkpoint_dir, to_image

from torch.utils.data import DataLoader
import data.make_patches as mp
from data.make_patches import pad_img, unpad_img
from utils.wavelet import swt2d, iswt2d, iswt2d_rgb, unnormalize_coeffs

from models import set_model

from utils.wavelet import standarize_coeffs, unstandarize_coeffs
from scipy.io.matlab.mio import savemat, loadmat
from data.images import ImageDataset, SWTImageDataset

def run_train(opt, training_data_loader):
    # check gpu setting with opt arguments
    opt = set_gpu(opt)

    print('Initialize networks for training')
    net = set_model(opt)
    print(net)

    if opt.use_cuda:
        net = net.to(opt.device)
    
    print("Setting Optimizer")
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
        print("===> Use Adam optimizer")

    if opt.resume:
        opt.start_epoch, net, optimizer = load_model(opt, net, optimizer=optimizer)
    else:
        set_checkpoint_dir(opt)

    if opt.multi_gpu:
        net = nn.DataParallel(net)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    log_file = os.path.join(opt.checkpoint_dir, opt.model + "_log.csv")
    opt_file = os.path.join(opt.checkpoint_dir, opt.model + "_opt.txt")
    
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # Create log file when training start
    if opt.start_epoch == 1:
        with open(log_file, mode='w') as f:
            f.write("epoch,train_loss,valid_loss\n")
        save_config(opt)

    data_loader = {
        'train': training_data_loader,
    }
    modes = ['train', 'valid']

    l2_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    if opt.use_cuda:
        l2_criterion = l2_criterion.to(opt.device)
        l1_criterion = l1_criterion.to(opt.device)

    if opt.content_loss == 'l2':
        content_loss_criterion = l2_criterion
    elif opt.content_loss == 'l1':
        content_loss_criterion = l1_criterion
    else:
        raise ValueError("Specify content loss correctly (l1, l2)")

    if opt.style_loss == 'l2':
        style_loss_criterion = l2_criterion
    elif opt.style_loss == 'l1':
        style_loss_criterion = l1_criterion
    else:
        raise ValueError("Specify style loss correctly (l1, l2)")

    if opt.ll_loss == 'l2':
        ll_loss_criterion = l2_criterion
    elif opt.ll_loss == 'l1':
        ll_loss_criterion = l1_criterion
    else:
        raise ValueError("Specify style loss correctly (l1, l2)")

    nc = opt.n_channels
    np.random.seed(1024)
    sq = np.arange(1024)
    np.random.shuffle(sq)

    for epoch in range(opt.start_epoch, opt.n_epochs):
        opt.epoch_num = epoch
        for phase in modes:
            if phase == 'train':
                total_loss = 0.0
                total_psnr = 0.0
                total_iteration = 0

                net.train()

                mode = "Training"
                print("*** %s ***" % mode)
                start_time = time.time()

                for iteration, batch in enumerate(data_loader[phase], 1):
                    # (_, x), (_, target) = batch[0], batch[1]
                    x, target = batch[0], batch[1]
                    x_img, target_img = batch[3], batch[4]
                    lr_approx = batch[5]
                
                    if opt.use_cuda:
                        x = x.to(opt.device)
                        target = target.to(opt.device)

                    optimizer.zero_grad()

                    # epoch_loss = 0.
                    with torch.set_grad_enabled(phase=='train'):
                        out = net(x)

                        # norm_target = normalize_coeffs(target, ch_min=opt.ch_min, ch_max=opt.ch_max)
                        std_target = standarize_coeffs(target, ch_mean=opt.ch_mean, ch_std=opt.ch_std)
                        # norm_out = normalize_coeffs(out, ch_min=opt.ch_min, ch_max=opt.ch_max)
                        std_out = standarize_coeffs(out, ch_mean=opt.ch_mean, ch_std=opt.ch_std)
                        
                        ll_target = std_target[:,0:nc,:,:]
                        ll_out = std_out[:,0:nc,:,:]
                        high_target = std_target[:,nc:,:,:]
                        high_out = std_out[:,nc:,:,:]
                            
                        # log_channel_loss(std_out, std_target, content_loss_criterion)
                        ll_content_loss = content_loss_criterion(ll_target, ll_out)
                        ll_style_loss = 0
                        # content_loss = content_loss_criterion(norm_target, norm_out)
                        high_content_loss = content_loss_criterion(high_target, high_out)
                        high_style_loss = 0

                        ll_loss = ll_content_loss + ll_style_loss
                        high_loss = high_content_loss + high_style_loss
                        epoch_loss = opt.ll_weight * ll_loss + (1 - opt.ll_weight) * high_loss

                        # L1 loss for wavelet coeffiecients
                        l1_loss = 0

                        total_loss += epoch_loss.item()

                        epoch_loss.backward()
                        optimizer.step()

                    mse_loss = l2_criterion(out, target)
                    psnr = 10 * math.log10(1 / mse_loss.item())
                    total_psnr += psnr



                    print("High Content Loss: {:5f}, High Style Loss: {:5f}, LL Content Loss: {:5f}, LL Style Loss:{:5f}".format(
                        high_content_loss, high_style_loss, ll_content_loss, ll_style_loss))
                    print("{} {:4f}s => Epoch[{}/{}]({}/{}): Epoch Loss: {:5f} High Loss: {:5f} LL Loss: {:5f} L1 Loss: {:5f} PSNR: {:5f}".format(
                        mode, time.time() - start_time, opt.epoch_num, opt.n_epochs, iteration, len(data_loader[phase]),
                        epoch_loss.item(), high_loss.item(), ll_loss.item(), l1_loss, psnr))
                        
                    total_iteration = iteration

                total_loss = total_loss / total_iteration
                total_psnr = total_psnr / total_iteration

                train_loss = total_loss
                train_psnr = total_psnr

            else :
                net.eval()
                mode = "Validation"
                print("*** %s ***" % mode)
                valid_loss, valid_psnr = run_valid(opt, net, content_loss_criterion, sq)
                scheduler.step(valid_loss)

        with open(log_file, mode='a') as f:
            f.write("%d,%08f,%08f,%08f,%08f\n" % (
                epoch,
                train_loss,
                train_psnr,
                valid_loss,
                valid_psnr
            ))

        save_checkpoint(opt, net, optimizer, epoch, valid_loss)


def run_valid(opt, net, content_loss_criterion, sq) : 
    start_time = time.time()
    total_psnr = 0.0
    total_loss = 0.0
    avg_psnr = 0.0
    avg_loss = 0.0
    mse_criterion = nn.MSELoss()

    if opt.n_channels == 1 :
        noisy_fn = 'siddplus_valid_noisy_raw.mat'
        noisy_key = 'siddplus_valid_noisy_raw'
        noisy_mat = loadmat(os.path.join(opt.test_dir, opt.dataset,  noisy_fn))[noisy_key]

        gt_fn = 'siddplus_valid_gt_raw.mat'
        gt_key = 'siddplus_valid_gt_raw'
        gt_mat = loadmat(os.path.join(opt.test_dir, opt.dataset,  gt_fn))[gt_key]

        n_im, h, w = noisy_mat.shape
    else :
        noisy_fn = 'siddplus_valid_noisy_srgb.mat'
        noisy_key = 'siddplus_valid_noisy_srgb'
        noisy_mat = loadmat(os.path.join(opt.test_dir, opt.dataset,  noisy_fn))[noisy_key]

        gt_fn = 'siddplus_valid_gt_srgb.mat'
        gt_key = 'siddplus_valid_gt_srgb'
        gt_mat = loadmat(os.path.join(opt.test_dir, opt.dataset,  gt_fn))[gt_key]

        n_im, h, w, c = noisy_mat.shape
    
    n_im = n_im//4

    for i in range(n_im):

        if opt.n_channels == 1 :
            noisy = np.reshape(noisy_mat[sq[i], :, :], (h, w))
            gt = np.reshape(gt_mat[sq[i], :, :], (h, w))
        else : 
            noisy = np.reshape(noisy_mat[sq[i], :, :, :], (h, w, c))
            gt = np.reshape(gt_mat[sq[i], :, :, :], (h, w, c))

        if opt.model == 'waveletdl' :
            img_patch_dataset = SWTImageDataset(opt, noisy)
            img_patch_dataloader = DataLoader(dataset=img_patch_dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=False)
            approx_list = img_patch_dataset.get_approx_list()
        else:
            img_patch_dataset = ImageDataset(opt, noisy)
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
            out_list = np.zeros((out.shape[0], out.shape[2], out.shape[3], opt.n_channels), dtype=noisy.dtype)
            out_list = out_list.squeeze()

            for ii in range(out.shape[0]):

                if opt.n_channels == 1:
                    inv_swt = iswt2d(approx_list[ii], out[ii], wavelet=opt.wavelet_func)
                    inv_swt[inv_swt > 1.0] = 1.0
                    inv_swt[inv_swt < 0] = 0
                    out_list[ii] = inv_swt
                else:
                    inv_swt = iswt2d_rgb(approx_list[ii], out[ii], wavelet=opt.wavelet_func)
                    # print("inv_swt.max():", np.amax(inv_swt))
                    # print("inv_swt.min():", np.amin(inv_swt))
                    inv_swt[inv_swt > 1.0] = 1.0
                    inv_swt[inv_swt < 0] = 0
                    inv_swt *= 255
                    inv_swt = inv_swt.astype(np.uint8)
                    out_list[ii] = inv_swt
                    
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

        out_img = torch.Tensor(out_img)
        gt = torch.Tensor(gt)

        loss = content_loss_criterion(out_img, gt)
        total_loss += loss.item()

        mse_loss = mse_criterion(gt, out_img)
        psnr = 10 * math.log10(1 / mse_loss.item())
        total_psnr += psnr

        print("Validation %.2fs => Image[%d/%d]: Loss: %.10f PSNR: %.5f" %
                (time.time() - start_time, i, n_im, loss.item(), psnr))

    avg_psnr = total_psnr/(n_im)
    avg_loss = total_loss/(n_im)

    print("Valid LOSS avg : {:.10f}\nValid PSNR avg : {:.5f}".format(avg_loss, avg_psnr))

    return avg_loss, avg_psnr