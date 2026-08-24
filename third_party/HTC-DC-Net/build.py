def get_model_and_optimizer(cfgs, test=False):
    if cfgs["model"] == "htcdc":
        from htcdc import UBins
        model = UBins(cfgs)
    else:
        raise NotImplementedError
    if test:
        return model
        
    if cfgs["optimizer"] == "Adam":
        from torch.optim import Adam
        optim = Adam
    elif cfgs["optimizer"] == "SGD":
        from torch.optim import SGD
        optim = SGD
    elif cfgs["optimizer"] == "Nadam":
        from torch.optim import NAdam
        optim = NAdam
    elif cfgs["optimizer"] == "AdamW":
        from torch.optim import AdamW
        optim = AdamW
    else:
        raise NotImplementedError
    
    # Get weight decay from config
    weight_decay = cfgs.get("weight_decay", 0.0)
    optimizer = optim(filter(lambda x: x.requires_grad, model.parameters()), lr=cfgs["lr"], weight_decay=weight_decay)
    
    return model, optimizer