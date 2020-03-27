from data import get_train_valid_dataloader, get_test_img_list, get_test_noisy_list, get_train_dataloader

import utils.trainer_swt as ST

import utils.prep_result as PR

from utils.saver import load_config
from options import args

if __name__ == '__main__':
    opt = args

    if opt.mode == 'train':
        print(opt)

        train_data_loader, valid_data_loader = get_train_valid_dataloader(opt)
        only_train_data_loader = get_train_dataloader(opt)

        ST.run_train(opt, only_train_data_loader)
        # ST.run_train(opt, train_data_loader, valid_data_loader)

    elif opt.mode == 'result_sidd':
        opt = load_config(opt)
        print(opt)
        PR.prep_Result(opt)
