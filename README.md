# <Track1-rawRGB>
you can download checkpoint : https://drive.google.com/drive/u/0/folders/1VUY4bKA70JDHyzV95FAkclCA4omSBnKY

# Train 
python main.py --dataset sidd1 --wavelet_func haar --ll_weight 0.2 --n_threads 8 --in_skip --n_resblocks 16 --attn

# Test
python main.py --mode result_sidd --resume_best --in_skip --patch_offset 19 --ensemble
>>Select directory that you want to load : 0

# sidd_raw directory path
train data : 
../../data/denoising/train/sidd1/HIGH/*.tiff
../../data/denoising/train/sidd1/LOW/*.tiff

valid & test data : 
../../data/denoising/test/sidd1/*.mat

checkpoint save path : 
./checkpoint/sidd1*/*.pth

[*]our best checkpoint path : 
./checkpoint/sidd_raw__best_checkpoint/models_epoch_0052_loss_0.00248895.pth

test result save path : 
./test-result/sidd_raw__best_checkpoint-patch_offset19-ensemble/


# <Track2-sRGB>
you can download checkpoint : https://drive.google.com/drive/u/0/folders/1VUY4bKA70JDHyzV95FAkclCA4omSBnKY

# Train 
python main.py --dataset sidd2 --n_resblocks 64 --n_threads 8

# Test
python main.py --mode result_sidd --resume_best --patch_offset 19 --ensemble
>>Select directory that you want to load : 1

# sidd_raw directory path
train data : 
../../data/denoising/train/sidd2/HIGH/*.tiff
../../data/denoising/train/sidd2/LOW/*.tiff

valid & test data : 
../../data/denoising/test/sidd2/*.mat

checkpoint save path : 
./checkpoint/sidd2*/*.pth

[*]our best checkpoint path : 
./checkpoint/sidd_srgb__best_checkpoint/models_epoch_0099_loss_0.13744611.pth

test result save path : 
./test-result/sidd_srgb__best_checkpoint-patch_offset19-ensemble/