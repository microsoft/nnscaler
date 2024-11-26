import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from torch.utils.tensorboard import SummaryWriter 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import json
import hashlib

from tqdm import trange
import torch

from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from CVPR2022_DaGAN.logger import Logger
from CVPR2022_DaGAN.frames_dataset import DatasetRepeater
import CVPR2022_DaGAN.modules.keypoint_detector as KPD
from CVPR2022_DaGAN.animate import animate
import CVPR2022_DaGAN.modules.generator as gen_module
from CVPR2022_DaGAN.modules.discriminator import MultiScaleDiscriminator
from model_full import  DiscriminatorFullModel_NNSCALER
import model_full as MODEL_FULL

from nnscaler.parallel import ComputeConfig, ReuseType, build_optimizer, parallelize
from dataset import VDataset

"""
Main modifications:
    1. build dummy inputs and ComputeConfig for parallization
    2. parallelize generator, discriminator, kp_detector
    3. build_optimizer for generator, discriminator, kp_detector
    4. after loss.backward(), need to sync_shard_grad() and scale_grads() according optimizers
Note:
    batchsize in training and validation and test should be the same.
"""


def init_seeds(cuda_deterministic=True):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn
    seed = 0 + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


def main(rank, world_size):
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    # parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
    #                     help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--use_depth",action='store_true',help='depth mode')
    parser.add_argument("--rgbd",action='store_true',help='rgbd mode')
    parser.add_argument("--kp_prior",action='store_true',help='use kp_prior in final objective function')

    # alter model
    parser.add_argument("--generator",required=True,help='the type of genertor')
    parser.add_argument("--kp_detector",default='KPDetector',type=str,help='the type of KPDetector')
    parser.add_argument("--GFM",default='GeneratorFullModel_NNSCALER',help='the type of GeneratorFullModel')
    
    parser.add_argument("--batchsize",type=int, default=-1,help='user defined batchsize')
    parser.add_argument("--kp_num",type=int, default=-1,help='user defined keypoint number')
    parser.add_argument("--kp_distance",type=int, default=10,help='the weight of kp_distance loss')
    parser.add_argument("--depth_constraint",type=int, default=0,help='the weight of depth_constraint loss')

    parser.add_argument("--name",type=str,help='user defined model saved name')

    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += opt.name


    print("Training...")

    device=torch.device("cuda",rank)
    torch.cuda.set_device(device)
    config['train_params']['loss_weights']['depth_constraint'] = opt.depth_constraint
    config['train_params']['loss_weights']['kp_distance'] = opt.kp_distance
    if opt.kp_prior:
        config['train_params']['loss_weights']['kp_distance'] = 0
        config['train_params']['loss_weights']['kp_prior'] = 10
    if opt.batchsize != -1:
        config['train_params']['batch_size'] = opt.batchsize
    if opt.kp_num != -1:
        config['model_params']['common_params']['num_kp'] = opt.kp_num

    # create generator
    generator = getattr(gen_module, opt.generator)(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)
    if opt.verbose:
        print(generator)

    # create discriminator
    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])

    discriminator.to(device)
    if opt.verbose:
        print(discriminator)
        
    kp_detector = getattr(KPD, opt.kp_detector)(**config['model_params']['kp_detector_params'],
                            **config['model_params']['common_params'])
    kp_detector.to(device)
    if opt.verbose:
        print(kp_detector)

    if config['backend_params']['backend_name'] == "nnscaler":
        # backend configs
        plan_ngpus = config["backend_params"]["plan_ngpus"]
        runtime_ngpus = config["backend_params"]["runtime_ngpus"]
        batchsize = config["train_params"]["batch_size"]  # per gpu batch size
        batchsize *= plan_ngpus
        frame_shape = config["dataset_params"]["frame_shape"]
        # create dummy input tensors for nnscaler graph tracing
        kp_detector_dummy_input = {"x": torch.randn(batchsize, 3, frame_shape[0], frame_shape[1])}

        generator_dummy_input = {
            'source_image': torch.randn(batchsize, 3, 256, 256),
            'kp_driving': {
                'value': torch.randn(batchsize, 15, 2),
                'jacobian': torch.randn(batchsize, 15, 2, 2)
            },
            'kp_source': {
                'value': torch.randn(batchsize, 15, 2),
                'jacobian': torch.randn(batchsize, 15, 2, 2)
            },
            'source_depth': torch.randn(batchsize, 1, 256, 256),
            'driving_depth': None   # torch.randn(batchsize, 1, 256, 256)
        }

        pyramide_generated = {
            'prediction_1': torch.randn(batchsize, 3, 256, 256),
            'prediction_0.5': torch.randn(batchsize, 3, 128, 128),
            'prediction_0.25': torch.randn(batchsize, 3, 64, 64),
            'prediction_0.125': torch.randn(batchsize, 3, 32, 32)
        }

        detached_kp_driving = {
            'value': torch.randn(batchsize, 15, 2),
            'jacobian': torch.randn(batchsize, 15, 2, 2)
        }

        compute_config = ComputeConfig(
            plan_ngpus, runtime_ngpus, use_zero=False, user_config={"batch_size": batchsize}
        )
        # parallelize models
        kp_detector = parallelize(
            kp_detector,
            kp_detector_dummy_input,
            'data', # autodist
            compute_config,
            reuse=ReuseType.MOO,
        )

        generator = parallelize(
            generator,
            generator_dummy_input,
            'data', # autodist
            compute_config,
            reuse=ReuseType.MOO,
        )

        discriminator = parallelize(
            discriminator,
            {"x": pyramide_generated, "kp": detached_kp_driving},
            'data', # autodist
            compute_config,
            reuse=ReuseType.MOO,
        )
    else:
        generator= torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
        generator = DDP(generator,device_ids=[rank],broadcast_buffers=False)

        discriminator= torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        discriminator = DDP(discriminator,device_ids=[rank],broadcast_buffers=False)

        kp_detector= torch.nn.SyncBatchNorm.convert_sync_batchnorm(kp_detector)
        kp_detector = DDP(kp_detector,device_ids=[rank],broadcast_buffers=False)

    generator.to(device)
    discriminator.to(device)
    kp_detector.to(device)

    dataset = VDataset(is_train=True)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    combined_str = json.dumps(config, sort_keys=True) + json.dumps(vars(opt), sort_keys=True)
    hashstr = hashlib.sha256(combined_str.encode('utf-8')).hexdigest()
    if rank == 0:
        writer = SummaryWriter(os.path.join(log_dir, 'tensorboard-logs', hashstr))
    else:
        writer = None

    if opt.mode == 'train':
        train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, rank, device, opt, writer)


def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, rank, device, opt, writer=None):
    """
    Steps:
        1. build optimizer for generator, discriminator, kp_detector
        2. build scheduler for generator, discriminator, kp_detector
        3. build train and valid DataLoader with DistributedSampler
        4. build generator_full and discriminator_full
        5. train and validate epoches
    """
    train_params = config['train_params']

    if config['backend_params']['backend_name'] == "nnscaler":
        optimizer_generator = build_optimizer(
            generator,
            torch.optim.Adam,
            lr=train_params['lr_generator'],
            betas=(0.5, 0.999),
        )
        optimizer_discriminator = build_optimizer(
            discriminator,
            torch.optim.Adam,
            lr=train_params['lr_discriminator'],
            betas=(0.5, 0.999),
        )
        optimizer_kp_detector = build_optimizer(
            kp_detector,
            torch.optim.Adam,
            lr=train_params['lr_kp_detector'],
            betas=(0.5, 0.999),
        )
        scale_factor = 1.0 / (config["backend_params"]["runtime_ngpus"] / config["backend_params"]["plan_ngpus"])
    else:
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
        optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params and train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=torch.cuda.device_count(), shuffle=True, rank=rank)
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], num_workers=16, sampler=sampler, drop_last=True)

    generator_full = getattr(MODEL_FULL, opt.GFM)(kp_detector, generator, discriminator, config, opt)
    discriminator_full = DiscriminatorFullModel_NNSCALER(kp_detector, generator, discriminator, config)
    test_dataset = VDataset(is_train=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=torch.cuda.device_count(), rank=rank)
    test_dataloader = DataLoader(test_dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=8, sampler=test_sampler)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        # logger.register_tensorboard_writer(writer)
        for epoch in trange(start_epoch, train_params['num_epochs']):
            #parallel
            sampler.set_epoch(epoch)
            total = len(dataloader)
            epoch_train_loss = 0
            generator.train(), discriminator.train(), kp_detector.train()
            with tqdm(total=total, position=rank, desc=f"Rank {rank}, Epoch {epoch}", leave=True) as par:
                for i,x in enumerate(dataloader):
                    x['source'] = x['source'].to(device)
                    x['driving'] = x['driving'].to(device)
                    losses_generator, generated = generator_full(x)
                    
                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)
                    loss.backward()
                    if config['backend_params']['backend_name'] == "nnscaler":
                        optimizer_generator.sync_shard_grad()
                        optimizer_kp_detector.sync_shard_grad()
                        optimizer_generator.scale_grads(scale_factor)
                        optimizer_kp_detector.scale_grads(scale_factor)
                    optimizer_generator.step()
                    optimizer_generator.zero_grad()
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()
                    epoch_train_loss+=loss.item()

                    if train_params['loss_weights']['generator_gan'] != 0:
                        optimizer_discriminator.zero_grad()
                        losses_discriminator = discriminator_full(x, generated)
                        loss_values = [val.mean() for val in losses_discriminator.values()]
                        loss = sum(loss_values)

                        loss.backward()
                        if config['backend_params']['backend_name'] == "nnscaler":
                            optimizer_discriminator.sync_shard_grad()
                            optimizer_discriminator.scale_grads(scale_factor)
                        optimizer_discriminator.step()
                        optimizer_discriminator.zero_grad()
                    else:
                        losses_discriminator = {}

                    losses_generator.update(losses_discriminator)
                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                    # for k,v in losses.items():
                    #     writer.add_scalar(k, v, total*epoch+i)
                    if rank == 0:
                        logger.log_iter(losses=losses)
                    par.update(1)
                    torch.distributed.barrier()
            epoch_train_loss = epoch_train_loss/total
            if (epoch + 1) % train_params['checkpoint_freq'] == 0:
                if rank == 0:
                    writer.add_scalar('epoch_train_loss', epoch_train_loss, epoch)
            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            if rank == 0:
                logger.log_epoch(epoch, {'generator': generator,
                                        'discriminator': discriminator,
                                        'kp_detector': kp_detector,
                                        'optimizer_generator': optimizer_generator,
                                        'optimizer_discriminator': optimizer_discriminator,
                                        'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
            generator.eval(), discriminator.eval(), kp_detector.eval()
            if (epoch + 1) % train_params['checkpoint_freq'] == 0:
                epoch_eval_loss = 0
                for i, data in tqdm(enumerate(test_dataloader), position=rank, desc=f"Rank {rank}", leave=True):
                    data['source'] = data['source'].cuda()
                    data['driving'] = data['driving'].cuda()
                    losses_generator, generated = generator_full(data) 
                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)
                    epoch_eval_loss+=loss.item()
                epoch_eval_loss = torch.tensor(epoch_eval_loss).cuda()
                gather_epoch_eval_loss = [torch.zeros_like(epoch_eval_loss) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gather_epoch_eval_loss, epoch_eval_loss)
                gathered_epoch_eval_loss = torch.mean(torch.tensor(gather_epoch_eval_loss)).item()
                epoch_eval_loss = gathered_epoch_eval_loss / len(test_dataloader)
                if rank == 0:
                    logger.log_iter({'epoch_eval_loss': epoch_eval_loss})


if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    init_seeds()
    main(rank, world_size)
    dist.destroy_process_group()

    