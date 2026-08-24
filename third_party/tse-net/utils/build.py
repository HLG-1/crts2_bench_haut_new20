import torch
from torch.optim import lr_scheduler

def get_model_and_optimizer(cfgs, test=False):
    if cfgs["model"] == "tsenet":
        from tsenet import TSENet
        model = TSENet(cfgs)
    else:
        NotImplementedError
        
    if test:
        return model
        
    if cfgs["optimizer"] == "Adam":
        from torch.optim import Adam
        optim = Adam
    else:
        NotImplementedError

    optimizer = optim(filter(lambda x: x.requires_grad, model.parameters()), lr=cfgs["lr"], weight_decay=1e-4)
    return model, optimizer

def get_scheduler(optimizer, cfgs):
    return None
