import os, glob
from importlib import import_module
# from torch.utils.data import dataloader
from torch.utils import data as D
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.mode = datasets[0].mode

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if args.mode == 'train':
            datasets = []
            for d in args.train_datasets:
                module_name = d
                m = import_module('data.' + module_name.lower())
                if module_name == 'mayo':
                    module_name = 'Mayo'
                datasets.append(getattr(m, module_name)(args, name=d))

            self.train_dataloader = DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        elif args.mode == 'test':
            self.test_dataloader = []
            for d in args.test_datasets:
                testset = getattr(m, 'Benchmark')(args, mode='test', name=d)
            

            self.test_dataloader.append(
                DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )


def get_module_attr(dataset):
    if dataset == 'sidd1':
        module_name = 'sidd'
        attr= 'SIDD1'
    elif dataset == 'sidd2':
        module_name = 'sidd'
        attr= 'SIDD2'
    print("module_name:", module_name)
    print("attr:", attr)
    return module_name, attr

def get_train_valid_dataloader(args):
    datasets = []
    
    module_name, attr = get_module_attr(args.dataset)
    m = import_module('data.' + module_name.lower())

    datasets.append(getattr(m, attr)(args, name=args.dataset))

    train_ds = MyConcatDataset(datasets)
    train_len = int(args.train_ratio * len(train_ds))
    valid_len = len(train_ds) - train_len
    train_dataset, valid_dataset = D.random_split(train_ds, lengths=[train_len, valid_len])
    
    print("Number of train dataset samples:", train_len)
    print("Number of valid dataset samples:", valid_len)
    print("Threading {}".format(args.n_threads))

    train_data_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=args.n_threads)
    valid_data_loader = DataLoader(dataset=valid_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=args.n_threads)

    return train_data_loader, valid_data_loader

def get_train_dataloader(args):

    datasets = []

    module_name, attr = get_module_attr(args.dataset)
    m = import_module('data.' + module_name.lower())

    datasets.append(getattr(m, attr)(args, name=args.dataset))

    train_ds = MyConcatDataset(datasets)
    train_len = len(train_ds)
    
    print("Number of train dataset samples:", train_len)
    print("Threading {}".format(args.n_threads))

    train_data_loader = DataLoader(dataset=train_ds,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=args.n_threads)

    return train_data_loader



def get_test_img_list(opt):
    img_list = [os.path.join(opt.img_dir, x) for x in os.listdir(opt.img_dir)]
    gt_img_list = [os.path.join(opt.gt_img_dir, x) for x in os.listdir(opt.gt_img_dir)]
    return img_list, gt_img_list

def get_test_noisy_list(opt):
    img_list = [os.path.join(opt.img_dir, x) for x in os.listdir(opt.img_dir)]
    # print(img_list)
    return img_list

# def get_test_img_list(opt):

#     opt.img_dir = r'../../data/denoising/train/mayo/quarter_{}mm/'.format(opt.thickness)
#     opt.gt_img_dir = r'../../data/denoising/train/mayo/full_{}mm/'.format(opt.thickness)

#     img_list = glob.glob(os.path.join(opt.img_dir,'*', '*.tiff'))
#     gt_img_list = glob.glob(os.path.join(opt.gt_img_dir, '*', '*.tiff'))

#     print(len(img_list))
#     # print(img_list)
#     print(len(gt_img_list))
#     # print(gt_img_list)
#     return img_list, gt_img_list