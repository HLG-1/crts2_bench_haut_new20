import torch
import argparse
import wandb
import os
from datetime import datetime
import time
from tqdm import tqdm
from itertools import cycle
from utils.commons import AverageMeter, UpdatableDict, convert_from_string, data_to_device, load_yaml, \
    fix_seed_for_reproducability, save_config
from utils.build import get_model_and_optimizer, get_scheduler
from dataloaders import get_train_val_dataloaders, get_pseudo_dataloaders


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--config', default=None, help='Specify a config file path')
    parser.add_argument('--exp_config', default=None, help='Specify an experiment config file path')
    parser.add_argument('--restore', action='store_true', help='Restore the run')
    parser.add_argument('--overfit', action='store_true', help='Overfit on small batches for debugging')
    args, unknown_raw = parser.parse_known_args()
    unknown = []
    for ur in unknown_raw:
        unknown.extend(ur.split("="))
    return args, unknown

def train_tse(cfgs, logger, train_dataloader, val_dataloader, unlabeled_dataloader, model, optimizer, checkpoint_name=None, scheduler=None):
    patience = int(cfgs.get("patience", cfgs["max_epochs"]))
    curr_patience = 0
    metric_names = cfgs.get("early_stopping", None)
    metric_modes = cfgs.get("early_stopping_mode", ['max'])
    
    student_metric = cfgs.get("student_metric", None)
    student_mode = cfgs.get("student_mode", "min")
    best_student_metric = 999 if student_mode=="min" else -999

    early_stopping = False
    if metric_names is not None:
        early_stopping = True
        best_metric_init = {}
        for metric_name, metric_mode in zip(metric_names, metric_modes):
            if metric_mode == "max":
                best_metric_init[metric_name] = -999
            else:
                best_metric_init[metric_name] = 999
    
    if cfgs["restore"]:
        chkpt_name = checkpoint_name if checkpoint_name is not None else 'checkpoint_last.pth.tar'
        chkpt_file = os.path.join(cfgs["experiment_dir"], chkpt_name)
        if os.path.exists(chkpt_file):
            chkpt = torch.load(chkpt_file)
            global_step = chkpt["step"]
            global_step_unlabeled = chkpt.get("step_unlabeled", 0)
            start_epoch = chkpt["epoch"]

            model.load_state_dict(chkpt["state_dict"])
            optimizer.load_state_dict(chkpt["optimizer"])
            if scheduler:
                scheduler.load_state_dict(chkpt["scheduler"])
            if cfgs["early_stopping"] is not None:
                curr_patience = chkpt.get("patience", 0)
                best_metric = chkpt.get("best_metric", best_metric_init)
        else:
            global_step = 0
            start_epoch = 0
            global_step_unlabeled = 0
            if cfgs["early_stopping"] is not None:
                best_metric = best_metric_init
                curr_patience = 0
    else:
        global_step = 0
        start_epoch = 0
        global_step_unlabeled = 0
        if cfgs["early_stopping"] is not None:
            best_metric = best_metric_init
            curr_patience = 0
    
    if curr_patience == patience:
        return

    pretrained = cfgs.get("pretrained", False)
    if pretrained:
        chkpt_file = os.path.join(cfgs["pretrained_dir"], f"checkpoint_{pretrained}.pth.tar")
        chkpt = torch.load(chkpt_file)
        if global_step <= chkpt["step"]:
            global_step = chkpt["step"]
            global_step_unlabeled = chkpt.get("step_unlabeled", 0)
            start_epoch = chkpt["epoch"]
            model.load_state_dict(chkpt["state_dict"])
            if cfgs["early_stopping"] is not None:
                curr_patience = chkpt.get("patience", 0)
                best_metric = chkpt.get("best_metric", best_metric_init)
    
    short = cfgs.get("short", True)
    update_each_iter = cfgs.get("update_each_iter", True)
    update_each_epoch = cfgs.get("update_each_epoch", False)
    unlabeled_from_epoch = cfgs.get("unlabeled_from_epoch", 0)
    assert update_each_iter + update_each_epoch == 1, "Update should be done either each iter or each epoch!"

    for epoch in range(start_epoch, int(cfgs["max_epochs"])):
        log_dict = {}
        loader = zip(train_dataloader, unlabeled_dataloader) if short else zip(cycle(train_dataloader), unlabeled_dataloader)# does not iterate over all unlabled data, but shuffle them and use only a small subset
        len_loader = len(train_dataloader) if short else len(unlabeled_dataloader)
        model.train()
        loss_train = AverageMeter()
        for (l_image_idx, l_image, l_gt), (u_image_idx, u_image, u_gt) in tqdm(loader, total=len_loader, desc=f"Epoch {epoch+1}: training ..."):
            global_step += 1
            log_flag = (global_step % int(cfgs["log_interval"]) == 0)

            l_image = data_to_device(l_image, device=cfgs["device"])
            l_gt = data_to_device(l_gt, device=cfgs["device"])
            u_image = data_to_device(u_image, device=cfgs["device"])
            u_gt = data_to_device(u_gt, device=cfgs["device"])

            l_losses, l_pred = model(l_image, l_gt)

            if epoch >= unlabeled_from_epoch:
                u_losses, u_pred = model.forward_unlabeled(u_image, u_gt)
                loss_total = l_losses["loss_total"] + float(cfgs.get("u_weight", 1)) * u_losses["loss_total"]
            else:
                loss_total = l_losses["loss_total"]

            loss_train.update(loss_total.item(), len(l_image)+len(u_image))

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            if update_each_iter:
                model.update_teacher()

            if log_flag:
                log_dict = {
                    'step': global_step,
                    'epoch': epoch + 1
                }
                
                log_dict.update({'train/'+key: loss for key, loss in l_losses.items()})
                log_dict.update(model.vis(l_image, l_pred, l_gt, image_idx=l_image_idx[0]))
                if epoch >= unlabeled_from_epoch:
                    log_dict.update({'train_unlabeled/'+key: loss for key, loss in u_losses.items()})
                    log_dict.update(model.vis(u_image, u_pred, u_gt, image_idx=u_image_idx[0], unlabeled=True))
                if scheduler is not None:
                    log_dict.update({"lr": float(optimizer.param_gropus[0]["lr"])})
                logger.log(log_dict)
        
        if update_each_epoch:
            model.update_teacher()
        model.update_rank()

        logger.log({
            'epoch': epoch + 1,
            'train/loss_avg': loss_train.avg
        })
        model.eval()

        loss_val = AverageMeter()
        eval_dict = UpdatableDict()
        with torch.no_grad():
            for image_idx, image, gt in tqdm(val_dataloader, desc=f"Epoch {epoch+1}: validating ..."):
                image = data_to_device(image, device=cfgs["device"])
                gt = data_to_device(gt, device=cfgs["device"])
                losses, pred, eval_params = model(image, gt)
                loss_val.update(losses["loss_total"].item(), len(image))
                eval_dict.update(eval_params)
        save_dict = {
            'step': global_step,
            'step_unlabeled': global_step_unlabeled,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if scheduler is not None:
            save_dict.update({
                "scheduler": scheduler.state_dict()
            })

        log_dict = {'epoch': epoch}
        log_dict.update({'val/loss_total': loss_val.avg})
        log_dict.update(model.vis(image, pred, gt, validation=True, image_idx=image_idx))
        
        eval_res = model.evaluate(eval_dict())
        log_dict.update(eval_res)

        if student_metric:
            if student_mode == "min":
                if eval_res[student_metric] < best_student_metric:
                    best_student_metric = eval_res[student_metric]

            if student_mode == "max":
                if eval_res[student_metric] > best_student_metric:
                    best_student_metric = eval_res[student_metric]

        if early_stopping:
            no_stop_flag = []
            for metric_name, metric_mode in zip(metric_names, metric_modes):
                if ((metric_mode == 'max') & (log_dict[metric_name] > best_metric[metric_name])) | \
                ((metric_mode == 'min') & (log_dict[metric_name] < best_metric[metric_name])):
                    best_metric[metric_name] = log_dict[metric_name]
                    no_stop_flag.append(True)
                    curr_patience = 0
                else:
                    no_stop_flag.append(False)

            if not any(no_stop_flag):
                curr_patience += 1
            else:
                save_dict.update({
                    'best_metric': best_metric,
                })
                for metric_name, no_stop in zip(metric_names, no_stop_flag):
                    if no_stop:
                        if '/' in metric_name:
                            metric_name = metric_name.split('/')[-1]
                        torch.save(save_dict, os.path.join(cfgs["experiment_dir"], 'checkpoint_best_{:s}.pth.tar'.format(metric_name)))
        
        save_dict.update({
            "patience": curr_patience
        })
        logger.log(log_dict)
        
        if (epoch % cfgs["checkpoint_interval"]) == 0:
            torch.save(save_dict, os.path.join(cfgs["experiment_dir"], 'checkpoint_{:03d}.pth.tar'.format(epoch))) 
        
        torch.save(save_dict, os.path.join(cfgs["experiment_dir"], 'checkpoint_last.pth.tar'))

        if early_stopping & (curr_patience == patience):
            print(f"Epoch {epoch+1}: maximum patience reached, early stopping ...")
            break
 
def train(cfgs, logger, train_dataloader, val_dataloader, model, optimizer, checkpoint_name=None, scheduler=None):
    patience = int(cfgs.get("patience", cfgs["max_epochs"]))
    curr_patience = 0
    metric_names = cfgs.get("early_stopping", None)
    metric_modes = cfgs.get("early_stopping_mode", ['max'])
    
    early_stopping = False
    if metric_names is not None:
        early_stopping = True
        best_metric_init = {}
        for metric_name, metric_mode in zip(metric_names, metric_modes):
            if metric_mode == "max":
                best_metric_init[metric_name] = -999
            else:
                best_metric_init[metric_name] = 999
    
    if cfgs["restore"]:
        chkpt_name = checkpoint_name if checkpoint_name is not None else 'checkpoint_last.pth.tar'
        chkpt_file = os.path.join(cfgs["experiment_dir"], chkpt_name)
        if os.path.exists(chkpt_file):
            chkpt = torch.load(chkpt_file)
            global_step = chkpt["step"]
            start_epoch = chkpt["epoch"]

            model.load_state_dict(chkpt["state_dict"])
            optimizer.load_state_dict(chkpt["optimizer"])
            if cfgs["early_stopping"] is not None:
                curr_patience = chkpt.get("patience", 0)
                best_metric = chkpt.get("best_metric", best_metric_init)
        else:
            global_step = 0
            start_epoch = 0
            if cfgs["early_stopping"] is not None:
                best_metric = best_metric_init
                curr_patience = 0
    else:
        global_step = 0
        start_epoch = 0
        if cfgs["early_stopping"] is not None:
            best_metric = best_metric_init
            curr_patience = 0
    
    if curr_patience == patience:
        return

    for epoch in range(start_epoch, int(cfgs["max_epochs"])):
        model.train()
        loss_train = AverageMeter()
        for image_idx, image, gt in tqdm(train_dataloader, desc=f"Epoch {epoch+1}: training ..."):
            global_step += 1
            log_flag = (global_step % int(cfgs["log_interval"]) == 0)

            image = data_to_device(image, device=cfgs["device"])
            gt = data_to_device(gt, device=cfgs["device"])
            losses, pred = model(image, gt)
            loss_total = losses["loss_total"]
            loss_train.update(loss_total.item(), len(image))

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            if log_flag:
                log_dict = {
                    'step': global_step,
                    'epoch': epoch
                }

                log_dict.update({'train/'+key: loss for key, loss in losses.items()})
                log_dict.update(model.vis(image, pred, gt, image_idx=image_idx))
                logger.log(log_dict)
        logger.log({
            'epoch': epoch + 1,
            'train/loss_avg': loss_train.avg
        })

        model.eval()
        loss_val = AverageMeter()
        eval_dict = UpdatableDict()
        with torch.no_grad():
            for image_idx, image, gt in tqdm(val_dataloader, desc=f"Epoch {epoch+1}: validating ..."):
                image = data_to_device(image, device=cfgs["device"])
                gt = data_to_device(gt, device=cfgs["device"])
                losses, pred, eval_params = model(image, gt)
                loss_val.update(losses["loss_total"].item(), len(image))
                eval_dict.update(eval_params)

        save_dict = {
            'step': global_step,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        log_dict = {'epoch': epoch}
        log_dict.update({'val/loss_total': loss_val.avg})
        log_dict.update(model.vis(image, pred, gt, True, image_idx=image_idx))

        if early_stopping:
            eval_res = model.evaluate(eval_dict())
            log_dict.update(eval_res)
            no_stop_flag = []
            for metric_name, metric_mode in zip(metric_names, metric_modes):
                if ((metric_mode == 'max') & (log_dict[metric_name] > best_metric[metric_name])) | \
                ((metric_mode == 'min') & (log_dict[metric_name] < best_metric[metric_name])):
                    best_metric[metric_name] = log_dict[metric_name]
                    no_stop_flag.append(True)
                    curr_patience = 0
                else:
                    no_stop_flag.append(False)

            if not any(no_stop_flag):
                curr_patience += 1
            else:
                save_dict.update({
                    'best_metric': best_metric,
                })
                for metric_name, no_stop in zip(metric_names, no_stop_flag):
                    if no_stop:
                        if '/' in metric_name:
                            metric_name = metric_name.split('/')[-1]
                        torch.save(save_dict, os.path.join(cfgs["experiment_dir"], 'checkpoint_best_{:s}.pth.tar'.format(metric_name)))
        save_dict.update({
            "patience": curr_patience
        })
        logger.log(log_dict)
        
        if (epoch % cfgs["checkpoint_interval"]) == 0:
            torch.save(save_dict, os.path.join(cfgs["experiment_dir"], 'checkpoint_{:03d}.pth.tar'.format(epoch))) 
        
        torch.save(save_dict, os.path.join(cfgs["experiment_dir"], 'checkpoint_last.pth.tar'))

        if early_stopping & (curr_patience == patience):
            print(f"Epoch {epoch+1}: maximum patience reached, early stopping ...")
            break

def main():
    args, unknown = parse_arguments()
    cfgs = {}

    if not args.restore:
        assert (args.config is not None) \
            & (args.exp_config is not None) \
            & os.path.isfile(args.config) \
            & os.path.isfile(args.exp_config), "Config files should be specified and exist"

        cfgs = load_yaml(args.config)
        cfgs.update(load_yaml(args.exp_config))
        cfgs["restore"] = False
        cfgs["overfit"] = args.overfit
        cfgs["checkpoint_dir"] = os.path.join(cfgs["checkpoint_dir"], cfgs["model"])
        cfgs["experiment_dir"] = os.path.join(cfgs["checkpoint_dir"], datetime.now().strftime('%y%m%d_%H%M%S'))
        if cfgs["overfit"]:
            cfgs["log_interval"] = 1
            cfgs["checkpoint_interval"] = cfgs["max_epochs"]
            cfgs["batch_size"] = 1
            cfgs["patience"] = cfgs["max_epochs"]
        
        print(f"Starting from {args.exp_config}...")

    else:
        assert (args.exp_config is not None) \
            & os.path.isfile(args.exp_config), "Experiment config file should be specified and exist"
        cfgs = load_yaml(args.exp_config)

        print(f"Restoring from {args.exp_config}...")
        cfgs['restore'] = True

    run_id = cfgs.get("wandb_run_id", None)

    if unknown:
        assert (len(unknown)%2==0), "Misc variables should be in pairs, key and value"
        for key, value in zip(unknown[0::2], unknown[1::2]):
            cfgs[key] = convert_from_string(value)
    
    
    project = cfgs.get("project1", 'unlabel4')
    runname = cfgs.get("name", None)
    try:
        logger = wandb.init(project=project, entity='chen_sn', id=run_id, resume='auto', name=runname)
    except wandb.errors.CommError as e:
        logger = wandb.init(project=project, entity="chen_sn", id=run_id+"_1", resume="auto", name=runname)
    cfgs["wandb_run_id"] = logger.id

    save_config(cfgs, os.path.join(cfgs["experiment_dir"], 'config.yaml'))
    logger.config.update(cfgs, allow_val_change=True)
    seed = int(cfgs.get("seed", 42))
    fix_seed_for_reproducability(seed)
 
    train_loader, val_loader = get_train_val_dataloaders(cfgs)
    model, optimizer = get_model_and_optimizer(cfgs)
    model.to(cfgs["device"])
    logger.watch(model)

    if cfgs["model"] == "tsenet":
        unlabeled_loader = get_pseudo_dataloaders(cfgs)
        train_tse(cfgs, logger, train_loader, val_loader, unlabeled_loader, model, optimizer)
    else:        
        train(cfgs, logger, train_loader, val_loader, model, optimizer)

if __name__ == "__main__":
    main()
