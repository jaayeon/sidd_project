import os
import datetime

import numpy as np
import torch
from utils.wavelet import iswt2d, iswt2d_rgb

def set_gpu(opt):
    if opt.use_cuda and torch.cuda.is_available():
        print("Setting GPU")
        print("===> CUDA Available: ", torch.cuda.is_available())
        opt.use_cuda = True
        opt.device = 'cuda'
    else:
        opt.use_cuda = False
        opt.device = 'cpu'

    if opt.use_cuda and torch.cuda.device_count() > 1 and opt.multi_gpu:
        print("===> Use " + str(torch.cuda.device_count()) + " GPUs")
        opt.multi_gpu = True
    else:
        opt.multi_gpu = False

    num_gpus = torch.cuda.device_count()
    opt.gpu_ids = []
    for id in range(num_gpus):
        opt.gpu_ids.append(id)
    print("GPU IDs:", opt.gpu_ids)

    return opt

def set_checkpoint_dir(opt):
    dt = datetime.datetime.now()
    date = dt.strftime("%Y%m%d")

    model_opt = opt.dataset + "-" + date + "-" + opt.model + '-patch' + str(opt.patch_size) + "-n_resblocks" + str(opt.n_resblocks)
    
    if opt.swt:
        model_opt = model_opt + "-swt-" + opt.wavelet_func + "-lv" + str(opt.swt_lv)
        if opt.parallel:
            model_opt = model_opt + "-parallel" + "-channels_per_group" +  str(opt.channels_per_group)
        else:
            model_opt = model_opt + "-serial" + "-n_feats" + str(opt.n_feats)

        model_opt = model_opt + "-cl_" + opt.content_loss + "-sl_" + opt.style_loss
    
    if opt.l1_weight != 0:
        model_opt = model_opt + '-l1_weight' + str(opt.l1_weight)
    

    if opt.attn and not opt.attn_par:
        model_opt = model_opt + '-attn' + '-spatial' + str(opt.spatial_attn) + '-reduction_ratio' + str(opt.reduction_ratio)
    elif opt.attn and opt.attn_par:
        model_opt = model_opt + '-attn_par' + '-spatial' + str(opt.spatial_attn) + '-reduction_ratio' + str(opt.reduction_ratio)
    
    if opt.attn:
        if 'avg' in opt.pool_type:
            model_opt = model_opt + 'avg'
        if 'max' in opt.pool_type:
            model_opt = model_opt + 'max'

    opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, model_opt)

def set_test_dir(opt):
    model_opt = os.path.basename(opt.checkpoint_dir)

    if opt.test_patches:
        test_dir_opt = model_opt + '-patch_offset' + str(opt.patch_offset)
    else:
        test_dir_opt = model_opt + "-image"

    if opt.ensemble:
        test_dir_opt = test_dir_opt + "-ensemble"
    opt.test_result_dir = os.path.join(opt.test_result_dir , test_dir_opt)


def to_image(opt, approx, swt_coeffs_tensor):
    # print("approx.shape:", approx.shape)
    approx = approx.numpy()
    bc, sw_nc, h, w = swt_coeffs_tensor.size()
    ret = torch.zeros((bc, opt.n_channels, h, w)).float()

    for i in range(bc):
        swt_coeff = swt_coeffs_tensor[i]
        if opt.use_cuda:
            swt_coeff = swt_coeff.detach().to('cpu').numpy()
        else:
            swt_coeff = swt_coeff.detach().numpy()

        if opt.n_channels == 1:
            img = iswt2d(approx[i], swt_coeff, wavelet=opt.wavelet_func)
            img = np.expand_dims(img, axis=2)
        elif opt.n_channels == 3:
            img = iswt2d_rgb(approx[i], swt_coeff, wavelet=opt.wavelet_func)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        ret[i] = img
    
    return ret