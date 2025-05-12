import os
import sys
import time
import h5py
import json
import torch
import pickle
import logging
import argparse
import cProfile
import numpy as np
# import matplotlib.pyplot as plt
from icecream import ic
from shutil import copyfile
from collections import OrderedDict
import torchvision
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torchsummary import summary
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel

from my_utils import logging_utils
logging_utils.config_logger()
from my_utils.YParams import YParams
from my_utils.darcy_loss import LossScaler, LpLoss, channel_wise_LpLoss
from my_utils.data_loader_multifiles import get_data_loader

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
import torch.utils.checkpoint as checkpoint
import gc



class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) 

    def __init__(self, params, world_rank):

        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        land_mask_path = os.path.join(script_dir, self.params.land_mask_path)

        with h5py.File(land_mask_path, 'r') as _f:
            self.mask_data = torch.as_tensor(_f['fields'])[0, self.params.out_channels].to(self.device, dtype=torch.bool)
       
        # Init gpu
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device('cuda', local_rank)
        logging.info('device: %s' % self.device)
        train_data_path = os.path.join(script_dir, params.train_data_path)
        valid_data_path = os.path.join(script_dir, params.valid_data_path)
        # Load data
        logging.info('rank %d, begin data loader init' % world_rank)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(
                params, 
                train_data_path,
                dist.is_initialized(),
                train=True)
        self.valid_data_loader, self.valid_dataset, self.valid_sampler = get_data_loader(
                params, 
                valid_data_path,
                dist.is_initialized(), 
                train=True)

        if params.loss_channel_wise:
            self.loss_obj = channel_wise_LpLoss(scale = params.loss_scale)
        else:
            self.loss_obj = LpLoss()

        # loss scaler
        self.mse_loss_scaler = LossScaler()

        logging.info('rank %d, data loader initialized' % world_rank)

        if params.nettype == 'Fourcastnet':
            from networks.Fourcastnet import Fourcastnet as model
        if params.nettype == 'Triton':
            from networks.Triton import Triton as model
        else:
            raise Exception("not implemented")
            
        self.model = model(params).to(self.device)


        if params.optimizer_type == 'FusedAdam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params.lr)

        if params.enable_amp == True:
            self.gscaler = amp.GradScaler()

        if dist.is_initialized():
            self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[params.local_rank],
                    output_device=[params.local_rank],
                    find_unused_parameters=False
            )

        self.iters = 0
        self.startEpoch = 0

        if (params.multi_steps_finetune == 1) and (params.resuming):
            logging.info("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)

        if params.multi_steps_finetune > 1:
            logging.info("Starting from pretrained one-step model at %s"%params.pretrained_ckpt_path)
            self.restore_checkpoint(params.pretrained_ckpt_path)
            self.iters = 0
            self.startEpoch = 0
            logging.info("Adding %d epochs specified in config file for refining pretrained model"%params.finetune_max_epochs)
            params['max_epochs'] = params.finetune_max_epochs

        self.epoch = self.startEpoch

        # Dynamical Learning rate
        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    factor=0.2, 
                    patience=5, 
                    mode='min'
            )
        elif params.scheduler == 'CosineAnnealingLR': 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=params.max_epochs,
                    last_epoch=self.startEpoch - 1
            )
        else:
            self.scheduler = None

        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)

            start = time.time()
            tr_time, data_time, step_time, train_logs = self.train_one_epoch() 
            valid_time, valid_logs = self.validate_one_epoch()

            if epoch == self.params.max_epochs - 1 and self.params.prediction_type == 'direct':
                valid_weighted_rmse = self.validate_final()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                    logging.info('lr for epoch {} is {}'.format(epoch + 1, self.optimizer.param_groups[0]['lr']))
                    logging.info('train data time={}, train per epoch time={}, train per step time={}, valid time={}'.format(data_time, tr_time, step_time, valid_time))
                    logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['train_loss'], valid_logs['valid_loss']))
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    exit()

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss'] <= best_valid_loss:
                        logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss']

            if self.params.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                logging.info('lr for epoch {} is {}'.format(epoch + 1, self.optimizer.param_groups[0]['lr']))
                logging.info('train data time={}, train per epoch time={}, train per step time={}, valid time={}'.format(data_time, tr_time, step_time, valid_time))
                logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['train_loss'], valid_logs['valid_loss']))

            torch.cuda.empty_cache()
            gc.collect()

    def land_mask_func(self, x, y):
        x = torch.masked_fill(input=x, mask=~self.mask_data, value=0)
        y = torch.masked_fill(input=y, mask=~self.mask_data, value=0)
        return x, y

    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()

        steps_in_one_epoch = 0
        for i, data in enumerate(self.train_data_loader, 0):
            self.iters += 1
            steps_in_one_epoch += 1 

            data_start = time.time()
            
            (inp, tar) = data

            data_time += time.time() - data_start

            tr_start = time.time()
            self.model.zero_grad()

            num_steps = params.multi_steps_finetune
            # print('num_steps:', num_steps)

            with amp.autocast(self.params.enable_amp):
                
                gen_prev = None
                loss = 0.0
                cw_loss = 0.0

                for step_idx in range(num_steps):
                    if step_idx == 0:
                        inp_step_1 = inp.to(self.device, dtype = torch.float32)
                        if params.multi_steps_finetune == 1:
                            gen_cur = self.model(inp_step_1)
                        else:
                            gen_cur = checkpoint.checkpoint(self.model, inp_step_1, use_reentrant=False)
                    else:
                        atmos_force = tar[:, step_idx, self.params.atmos_channels].to(self.device, dtype=torch.float)
                        gen_prev = torch.cat( (gen_prev, atmos_force), axis = 1).to(self.device, dtype = torch.float32)
                        gen_cur = checkpoint.checkpoint(self.model, gen_prev, use_reentrant=False)

                    if params.multi_steps_finetune == 1:
                        tar_step = tar[:, self.params.out_channels].to(self.device, dtype=torch.float)
                    else:
                        tar_step = tar[:, step_idx, self.params.out_channels].to(self.device, dtype=torch.float)
                        
                    if self.params.land_mask:
                        # print('land_mask')
                        gen_cur, tar_step = self.land_mask_func(gen_cur, tar_step)
                        
                    if self.params.use_loss_scaler_from_metnet3:
                        gen_cur = self.mse_loss_scaler(gen_cur)

                    loss_step, cw_loss_step = self.loss_obj(gen_cur, tar_step)

                    loss += loss_step
                    cw_loss += cw_loss_step
                    if step_idx == 0:
                        del inp
                        mse1 = torch.mean((gen_cur - tar_step) ** 2).item()

                    gen_prev = gen_cur

                    del tar_step, gen_cur
                del gen_prev
                
            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()
            print('1_step_mse:', mse1)
                

            if self.params.enable_amp:
                self.gscaler.update()
            break

            tr_time += time.time() - tr_start

        logs = {'train_loss': loss}

        for vi, v in enumerate(self.params.out_variables):
            logs[f'{v}_train_loss'] = cw_loss[vi]

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())

        # time of one step in epoch
        step_time = tr_time / steps_in_one_epoch

        return tr_time, data_time, step_time, logs

    def validate_one_epoch(self):

        logging.info('validating...')
        self.model.eval()

        valid_buff  = torch.zeros((3+self.params.N_out_channels), dtype=torch.float32, device=self.device)
        valid_loss  = valid_buff[0].view(-1) # 0
        valid_l1    = valid_buff[1].view(-1) # 0
        valid_steps = valid_buff[-1].view(-1) # 0

        valid_start = time.time()
        sample_idx = np.random.randint(len(self.valid_data_loader))
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                if i > 1:
                    break
                inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
                # gen = self.model(inp)
                num_steps = params.multi_steps_finetune
                for step_idx in range(num_steps):
                    if step_idx == 0:
                        inp_step_1 = inp.to(self.device, dtype = torch.float32)
                        gen_cur = self.model(inp_step_1)
                    else:
                        atmos_force = tar[:, step_idx, self.params.atmos_channels].to(self.device, dtype=torch.float)
                        gen_prev = torch.cat( (gen_prev, atmos_force), axis = 1).to(self.device, dtype = torch.float32)
                        gen_cur = self.model(gen_prev)
                        
                    if params.multi_steps_finetune == 1:
                        tar_step = tar[:, self.params.out_channels].to(self.device, dtype=torch.float)
                    else:
                        tar_step = tar[:, step_idx, self.params.out_channels].to(self.device, dtype=torch.float)
                    if self.params.land_mask:
                        gen_cur, tar_step = self.land_mask_func(gen_cur, tar_step)
                    if step_idx == 0:
                        del inp_step_1
                    gen_prev = gen_cur

                    if step_idx == params.multi_steps_finetune - 1:
                        gen, tar = gen_cur, tar_step
                        
                    del tar_step, gen_cur
                del gen_prev
                    
                gen.to(self.device, dtype=torch.float)

                if self.params.land_mask:
                    gen, tar = self.land_mask_func(gen, tar)

                _, cw_valid_loss = self.loss_obj(gen, tar)
                valid_loss_ = torch.mean((gen[:, :, :, :] - tar[:, :, :, :]) ** 2).item()
                valid_loss += valid_loss_
                valid_l1   += nn.functional.l1_loss(gen, tar)

                for vi, v in enumerate(self.params.out_variables):
                    valid_buff[vi+2] += cw_valid_loss[vi]

                valid_steps += 1.

                # save fields for vis before log norm
                os.makedirs(params['experiment_dir'] + "/" + str(i), exist_ok =True)
                
                del gen, tar

        if dist.is_initialized():
            dist.all_reduce(valid_buff)

        # divide by number of steps
        valid_buff[0:-1] = valid_buff[0:-1] / valid_buff[-1] # loss/steps, l1/steps
        valid_buff_cpu = valid_buff.detach().cpu().numpy()

        valid_time = time.time() - valid_start
        
        logs = {'valid_loss': valid_buff_cpu[0],
                'valid_l1':   valid_buff_cpu[1]}
        for vi, v in enumerate(self.params.out_variables):
            logs[f'{v}_valid_loss'] = valid_buff_cpu[vi+2]

    
        return valid_time, logs

    def load_model(self, model_path):
        if self.params.log_to_screen:
            logging.info('Loading the model weights from {}'.format(model_path))

        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.params.local_rank))

        if dist.is_initialized():
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            new_model_state = OrderedDict()
            model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'
            for key in checkpoint[model_key].keys():
                if 'module.' in key:  # model was stored using ddp which prepends module
                    name = str(key[7:])
                    new_model_state[name] = checkpoint[model_key][key]
                else:
                    new_model_state[key] = checkpoint[model_key][key]
            self.model.load_state_dict(new_model_state)
            self.model.eval()

    def save_checkpoint(self, checkpoint_path, model=None):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """

        if not model:
            model = self.model

        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        if self.params.resuming and (self.params.multi_steps_finetune == 1):
        # restore checkpoint is used for finetuning as well as resuming. 
        # If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/Model.yaml', type=str)  
    parser.add_argument("--multi_steps_finetune", default=1, type=int)  
    parser.add_argument("--finetune_max_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--config", default='AFNO', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--local_rank", default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, args.yaml_config)
    params = YParams(os.path.abspath(yaml_path), args.config, True)
    params['epsilon_factor'] = args.epsilon_factor
    params['multi_steps_finetune'] = args.multi_steps_finetune
    params['finetune_max_epochs']  = args.finetune_max_epochs

    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE']) # 进程组中的进程数
    print('world_size :', params['world_size'])

    print('Initialize distributed process group...')
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    params['local_rank'] = local_rank  # GPU ID

    torch.backends.cudnn.benchmark = True
    world_rank = dist.get_rank() # 获取当进程组中的进程号

    params['global_batch_size'] = args.batch_size
    params['batch_size'] = int(args.batch_size // params['world_size'])  # batch size must be divisible by the number of gpu's
    params['enable_amp'] = args.enable_amp  # Automatic Mixed Precision Training

    # Set up directory
    if params['multi_steps_finetune'] > 1:
        pretrained_expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
        multi_steps = params['multi_steps_finetune']
        if params['multi_steps_finetune'] > 2:
            params['pretrained_ckpt_path'] = os.path.join(pretrained_expDir, f'{multi_steps-1}_steps_finetune/training_checkpoints/best_ckpt.tar')
        else:
            params['pretrained_ckpt_path'] = os.path.join(pretrained_expDir, 'training_checkpoints/best_ckpt.tar')

        expDir = os.path.join(pretrained_expDir, f'{multi_steps}_steps_finetune')
        if world_rank == 0:
            os.makedirs(expDir, exist_ok=True)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok=True)

        params['experiment_dir'] = os.path.abspath(expDir)
        params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar') 
        params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

        params['resuming'] = True
    else:
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
        if world_rank == 0:
            os.makedirs(expDir, exist_ok =True)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok =True)
            copyfile(os.path.abspath(args.yaml_config), os.path.join(expDir, 'config.yaml'))

        params['experiment_dir'] = os.path.abspath(expDir)
        params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar') 
        params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

        # Do not comment this line out please:
        args.resuming = True if os.path.isfile(params.checkpoint_path) else False
        params['resuming'] = args.resuming

  
    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'train.log'))
        logging_utils.log_versions()
        params.log()

    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    params['in_channels'] = np.array(params['in_channels'])
    params['out_channels'] = np.array(params['out_channels'])
    params['N_out_channels'] = len(params['out_channels'])
    params['N_in_channels'] = len(params['in_channels']) 

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params, world_rank)
    trainer.train()
    logging.info('DONE ---- rank %d' % world_rank)
