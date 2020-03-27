import argparse
import os
import torch

data_dir = r'../../data/denoising'
src_dir = r'./'

train_dir = os.path.join(data_dir, 'train')
checkpoint_dir = os.path.join(src_dir, 'checkpoint')
test_dir = os.path.join(data_dir, 'test')
test_result_dir = os.path.join(src_dir, 'test-results')

parser = argparse.ArgumentParser(description='Denoising models')

parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'result_sidd'],
                    help='Specify the mode (train, result_sidd)')

parser.add_argument('--model', type=str, default='waveletdl',
                    choices=[
                        'waveletdl'
                    ],
                    help='Model ')

# Hardware specifications
parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help='Use multiple GPUs')
parser.add_argument('--no_multi_gpu', dest='multi_gpu', action='store_false',
                    help='Do not enable multiple GPUs')
parser.set_defaults(multi_gpu=True)
parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                    help='Use cuda')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false',
                    help='Do not use cuda')
parser.set_defaults(use_cuda=True)
parser.add_argument('--device', type=str, default='cpu',
                    help='CPU or GPU')
parser.add_argument("--n_threads", type=int, default=6,
                    help="Number of threads for data loader to use, Default: 8")
parser.add_argument('--n_cpu', type=int, default=8,
                    help='Number of cpu threads to use during batch generation')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')

# Image specifications
parser.add_argument('--patch_size', type=int, default=80,
                    help='Size of patch')
parser.add_argument('--patch_offset', type=int, default=5,
                    help='Size of patch offset')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB or pixel value')
parser.add_argument('--pixel_range', type=float, default=1.0,
                    help='maximum value of RGB or pixel value')
parser.add_argument('--n_channels', type=int, default=1, choices=[1, 3],
                    help='Number of image channels')
# parser.add_argument('--out_channels', type=int, default=1,
#                     help='Number of image channels of result')
parser.add_argument("--train_ratio", type=float, default=0.95,
                    help="Ratio of train dataset (ex: train:validation = 0.95:0.05), Default: 0.95")
parser.add_argument('--ch_min', nargs='+', default=[],
                    help='Minimun value of each channel')
parser.add_argument('--ch_max', nargs='+', default=[],
                    help='Maximun value of each channel')
parser.add_argument('--ch_mean', nargs='+', default=[],
                    help='Mean of each channel in wavelet transform')
parser.add_argument('--ch_std', nargs='+', default=[],
                    help='Standard deviation of each channe in wavelet transform')
parser.add_argument('--dwt_mean', nargs='+', default=[],
                    help='Mean of each channel in wavelet transform')
parser.add_argument('--dwt_std', nargs='+', default=[],
                    help='Standard deviation of each channe in wavelet transform')

# Data specifications
parser.add_argument('--dataset', type=str, default='sidd1')
parser.add_argument('--train_datasets', nargs='+', default=['sidd1'],
                    choices=['sidd1', 'sidd2'],
                    help='Specify dadtaset name (mayo or genoray)')

parser.add_argument('--ext', type=str, default='sep',
                    help='File extensions')
parser.add_argument('--data_dir', type=str, default=data_dir,
                    help='Path of training directory contains both lr and hr images')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')

parser.add_argument('--train_dir', type=str, default=train_dir,
                    help='Path of training directory contains both lr and hr images')
parser.add_argument('--test_dir', type=str, default=test_dir,
                    help='Path of directory to be tested (no ground truth)')
parser.add_argument('--use_npy', default=False, action='store_true',
                    help="Use npy files to load whole data into memory, Default: False")
parser.add_argument('--in_mem', default=False, action='store_true',
                    help="Load whole data into memory, Default: False")
parser.add_argument('--path_opt', default='', type=str,
                    help="Specify options in path name")
parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir,
                    help='Path to checkpoint directory')
parser.add_argument('--select_checkpoint', default=False, action='store_true',
                    help='Choose checkpoint directory')
parser.add_argument('--noise_train', default=False, action='store_true',
                    help='Train model for noise than generating clean image')
parser.add_argument('--augment', dest='augment', action='store_true',
                    help='Do random flip (vertical, horizontal, rotation)')
parser.add_argument('--no_augment', dest='augment', action='store_false',
                    help='Do not random flip (vertical, horizontal, rotation)')
parser.set_defaults(augment=True)

# Wavelet deep learning model specification
parser.add_argument('--swt', dest='swt', action='store_true',
                    help='Do level stationary wavelet transform')
parser.add_argument('--in_skip', default=False, action='store_true')
parser.add_argument('--no_swt', dest='swt', action='store_false',
                    help='Do level stationary wavelet transform')
parser.set_defaults(swt=False)
parser.add_argument('--wavelet_func', type=str, default='bior2.2',
                    help='Specify wavelet function')
parser.add_argument('--swt_lv', type=int, default=2,
                    help='Level of stationary wavelet transform')
parser.add_argument('--swt_num_channels', type=int, default=7,
                    help='Number of stationay wavelet transformed channels')
parser.add_argument('--parallel', dest='parallel', action='store_true',
                    help='Run parellel model for wavelet deep learning model')
parser.add_argument('--serial', dest='parallel', action='store_false',
                    help='Run serial model for wavelet deep learning model')
parser.set_defaults(parallel=False)

# EDSR, waveletdl
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
# EDSR, waveletdl
parser.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
# EDSR, waveletdl, ERDNet
parser.add_argument('--n_feats', type=int, default=96,
                    help='number of feature maps')
# EDSR, waveletdl
parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size of convolution layer')
# EDSR, waveletdl
parser.add_argument('--stride', type=int, default=1,
                    help='stride of convolution and deconvolution layers')
# EDSR, waveletdl
parser.add_argument('--bn', default=False, action='store_true',
                    help='Do batch normalization')
# EDSR, RCAN
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')

parser.add_argument('--bias', default=False, action='store_true',
                    help='Add bias')

# RIDNet, RCAN
parser.add_argument('--reduction', type=int, default=16,
                    help='Reduction rate')

#concat spatial domain img
parser.add_argument('--il_weight', type=float, default=0.1,
                    help='img_loss_weight(l1)')

# Attention map
parser.add_argument('--attn', default=False, action='store_true',
                    help='Use CBAM')
parser.add_argument('--attn_par', default=False, action='store_true',
                    help='Use paralel CBAM')
parser.add_argument('--spatial_attn', default=False, action='store_true',
                    help='Use spatial gate in attention')
parser.add_argument('--channel_attn', default=False, action='store_true',
                    help='Use channel gate in attention')
parser.add_argument('--reduction_ratio', type=int, default=16,
                    help='Reduction ratio in attention')
parser.add_argument('--pool_type', nargs='+', default=['avg'], choices=['avg', 'max'],
                    help='avg, max', )

parser.add_argument('--each_attn', type=int, default=4,
                    help='Add CBAM for each number of resblocks')
parser.add_argument('--n_attnblocks', type=int, default=8,
                    help='Number of attention blocks')

# Learning specification
parser.add_argument('--n_epochs', type=int, default=5000,
                    help='Number of epochs of training')
parser.add_argument('--resume', default=False, action='store_true',
                    help='Resume checkpoint')
parser.add_argument('--epoch_num', type=int, default=0,
                    help='epoch number to restart')
parser.add_argument('--resume_best', default=False, action='store_true',
                    help='Resume the best model from checkpoint')
parser.add_argument('--resume_last', default=False, action='store_true',
                    help='Resume the last model of epoch from checkpoint')


# Loss specification
parser.add_argument("--img_loss", type=str, default='l1',
                    help="Loss function (l1, l2)")
parser.add_argument('--content_loss', type=str, default='l1', choices=['l1', 'l2'],
                    help='Loss function (l1, l2)')
parser.add_argument('--style_loss', type=str, default='l1', choices=['l1', 'l2'],
                    help='Loss function (l1, l2)')
parser.add_argument('--ll_loss', type=str, default='l1', choices=['l1', 'l2'],
                    help='Loss function (l1, l2)')

parser.add_argument('--perceptual_loss', default=False, action='store_true',
                    help='Use perceptual loss')
parser.add_argument('--gram_matrix', default=False, action='store_true',
                    help='Do Gram Matrix')
parser.add_argument('--content_weights',  nargs='+', type=float, default=[],
                    help='content alpha ratio')
parser.add_argument('--style_weights',  nargs='+', type=float, default=[],
                    help='content beta ratio')
parser.add_argument('--ll_content_weights',  nargs='+', type=float, default=[],
                    help='content alpha ratio')
parser.add_argument('--ll_style_weights',  nargs='+', type=float, default=[],
                    help='content beta ratio')
parser.add_argument('--l1_weight', type=float, default=0.,
                    help='Content gamma ratio (L1 weight)')
parser.add_argument('--ll_weight', type=float, default=0.5,
                    help='Weight of LL loss to high frequency loss (all h) when perceptual loss is enabled')

# Optimizer specification
parser.add_argument("--optimizer", type=str, default='adam',
                    help="Loss function (adam, sgd)")
parser.add_argument('--lr', type=float, default=0.0002,
                    help='Adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9,
                    help='Adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='Adam: decay of second order momentum of gradient')
parser.add_argument("--start_epoch", type=int, default=1,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--batch_size', type=int, default=32,
                    help='Size of the batches')

# Result specifications
parser.add_argument("--ensemble", default=False, action='store_true',
                    help="Do self ensemble")
parser.add_argument("--save_test", default=False, action='store_true',
                    help="Save validation images for test")
parser.add_argument("--img_dir", type=str, default=None,
                    help="path to image directory")
parser.add_argument('--gt_img_dir', type=str, default=None,
                    help='Path to ground truth image directory')
parser.add_argument("--test_result_dir", type=str, default=test_result_dir,
                    help="Path to save result images")
parser.add_argument("--test_patches", dest='test_patches', action='store_true',
                    help="Divide image into patches")
parser.add_argument('--test_image', dest='test_patches', action='store_false',
                    help='Test whole image instead of dividing into patches')
parser.set_defaults(test_patches=True)
parser.add_argument('--report_swt', default=False, action='store_true',
                    help='Report swt analysis')

args = parser.parse_args()

if args.dataset == 'sidd2':
    args.n_channels = 3

if args.model == 'waveletdl' :
    args.swt = True
    # args.input_nc = args.swt_num_channels
    # args.output_nc = args.swt_num_channels
    args.input_nc = args.n_channels
    args.output_nc = args.n_channels

if args.swt:
    lv = args.swt_lv
    args.swt_num_channels = (3 * lv + 1) * args.n_channels
    # args.in_channels = args.swt_num_channels
    # args.out_channels = args.swt_num_channels

"""
If layer is [] and weight is [0], it will return 0
If layer is [], it will use original input before feature extractor
"""

if args.perceptual_loss:
    if not args.style_layers:
        raise ValueError("Please specify style layers when perceptual loss is enabled")
    if args.loss_type is None:
        args.loss_type = 'costum'
else:
    args.content_weights = [1.0]
    args.style_weights = [0]
    args.ll_content_weights = [1.0]
    args.ll_style_weights = [0]

if args.ensemble and args.mode == 'train':
    args.ensemble = False
    
if not args.ch_mean:
    if args.dataset == 'sidd1':
        args.ch_mean = [0.31146, 0, 0, 0, 0, 0, 0]
    elif args.dataset == 'sidd2':
        args.ch_mean = [1.2816,  1.0384,  0.96931, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        raise ValueError("Please specify ch_mean")
        # args.ch_mean = [2.0, 2.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # args.ch_mean = [0.439, 0.502, 0.505, 0.510, 0.493, 0.484, 0.504]

if not args.ch_std:
    if args.dataset == 'sidd1':
        args.ch_std = [0.0769, 0.0243, 0.0244, 0.0175, 0.0218, 0.0218, 0.0483]
    elif args.dataset == 'sidd2':
        args.ch_std = [0.2716, 0.1976, 0.2471, 0.1125, 0.0754, 0.1112,
                        0.1125, 0.0754, 0.1112, 0.0732, 0.0544, 0.0739,
                        0.0394, 0.0328, 0.0406, 0.0394, 0.0328, 0.0406,
                        0.0177, 0.0159, 0.0197]
    else:
        raise ValueError("Please specify ch_std")

if not args.ch_min:
    args.ch_min = [-1.9, -2.50, -2.60, -1.36, -1.0, -1.0, -0.40]
if not args.ch_max:
    args.ch_max = [5.72, 2.50, 2.55, 1.27, 1.10, 1.10, 0.40]

if args.dataset == 'sidd1':
    args.img_dir = r'../../data/denoising/test/sidd1/images/LOW'
    args.gt_img_dir = r'../../data/denoising/test/sidd1/images/HIGH'
elif args.dataset == 'sidd2':
    args.img_dir = r'../../data/denoising/test/sidd2/images/LOW'
    args.gt_img_dir = r'../../data/denoising/test/sidd2/images/HIGH'
else:
    raise ValueError('Please specify dataset!')

args.train_datasets = [args.dataset]

torch.manual_seed(args.seed)


