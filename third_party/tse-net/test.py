import torch
import argparse
import os
from tqdm import tqdm
from glob import glob
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

from utils.commons import AverageMeter, UpdatableDict, convert_from_string, data_to_device, load_yaml, \
    fix_seed_for_reproducability, save_config
from utils.build import get_model_and_optimizer
from dataloaders import get_test_dataloaders

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test configuration')
    parser.add_argument('--config', default=None, help='Specify a config file path')
    parser.add_argument('--vis', action='store_true', help='Visualize test results')
    parser.add_argument('misc', nargs='*', metavar='misc', help='Other variables')
    args = parser.parse_args()
    return args

def test(cfgs, test_dataloaders, model, logger):
    model.eval()
    chkpt_file = cfgs.get("test_checkpoint_file", 'checkpoint_best_rmse.pth.tar')
    result_files = glob(os.path.join(cfgs["experiment_dir"], chkpt_file.replace("checkpoint", "result")))
    if len(result_files) > 0:
        result_dict = torch.load(result_files[0])
        logger.summary['test'] = result_dict
        # return 
    chkpt_files = glob(os.path.join(cfgs["experiment_dir"], chkpt_file))
    if len(chkpt_files) == 0:
        chkpt_files = [os.path.join(cfgs["experiment_dir"], 'checkpoint_last.pth.tar')]

    for chkpt_file in chkpt_files:
        filename = os.path.basename(chkpt_file)
        chkpt = torch.load(chkpt_file, map_location=cfgs["device"])
        epoch = chkpt["epoch"]
        model.load_state_dict(chkpt["state_dict"])

        save_dict = {'epoch': epoch}
        with torch.no_grad():
            for data_name, test_dataloader in test_dataloaders.items():
                loss_test = AverageMeter()
                eval_dict = UpdatableDict()
                for _, image, gt in tqdm(test_dataloader, desc=f"Epoch {epoch+1}, {data_name}: testing ..."):
                    image = data_to_device(image, device=cfgs["device"])
                    gt = data_to_device(gt, device=cfgs["device"])

                    losses, pred, eval_params = model(image, gt)

                    loss_test.update(losses["loss_total"].item(), len(image))
                    eval_dict.update(eval_params)

                eval_res = model.evaluate(eval_dict())
                save_dict.update({
                    data_name: {
                        ** eval_res, 
                        'loss_total':loss_test.avg
                    }})
                
        save_dict = data_to_device(save_dict, 'cpu_test')
        logger.summary['test'] = save_dict
        torch.save(save_dict, os.path.join(cfgs["experiment_dir"], filename.replace('checkpoint', 'result')))

def main():
    args = parse_arguments()
    cfgs = {}
    assert (args.config is not None) \
        & os.path.isfile(args.config), "Config file should be specified and exist"
    cfgs = load_yaml(args.config)
    cfgs["vis_test"] = args.vis
    cfgs["test"] = True
    cfgs["experiment_dir"] = os.path.dirname(args.config)
    if args.misc:
        assert (len(args.misc)%2==0), "Misc variables should be in pairs, key and value"
        for key, value in zip(args.misc[0::2], args.misc[1::2]):
            cfgs[key] = convert_from_string(value)
    if "exp_config" in cfgs:
        cfgs.update(load_yaml(cfgs["exp_config"]))
    print(f"Starting test from {args.config}...")
    
    seed = int(cfgs.get("seed", 42))
    fix_seed_for_reproducability(seed)

    test_loaders = get_test_dataloaders(cfgs)

    project = cfgs.get("project1", 'unlabel4')
    model = get_model_and_optimizer(cfgs, True)
    logger = wandb.init(project=project, entity='chen_sn', id=cfgs["wandb_run_id"])
    model.to(cfgs["device"])

    test(cfgs, test_loaders, model, logger)


if __name__ == "__main__":
    main()
