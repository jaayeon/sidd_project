from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_model(opt):
    if opt.model == 'waveletdl':
        module_name = 'models.wavelet_dl'
    else:
        raise ValueError("Need to specify model (redcnn, dncnn)")
    
    module = import_module(module_name)
    model = module.make_model(opt)

    return model
