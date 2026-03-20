import argparse
import logging
import os
import pprint
import random
import time

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.builder import build_scalematch_model_from_cfg, load_backbone_weights
from supervised import evaluate
from util.classes import CLASSES
from util.dist_helper import setup_distributed
from util.ohem import ProbOhemCrossEntropy2d
from util.train_utils import (
    DictAverageMeter,
    confidence_weighted_loss,
    cutmix_img_,
    cutmix_mask,
)
from util.utils import count_params, init_log


DEFAULT_IMG_SCALES = [0.5, 0.75, 1.0, 1.25]
DEFAULT_FEAT_S_SCALES = [0.75]
DEFAULT_FEAT_L_SCALES = [1.0, 1.25, 1.5]
DEFAULT_WARM_UP = 10


parser = argparse.ArgumentParser(description='ScaleMatch for UniMatch-V2')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def get_scalematch_recipe(cfg):
    return {
        'img_scales': cfg.get('img_scales', DEFAULT_IMG_SCALES),
        'feat_s_scales': cfg.get('feat_s_scales', DEFAULT_FEAT_S_SCALES),
        'feat_l_scales': cfg.get('feat_l_scales', DEFAULT_FEAT_L_SCALES),
        'conf_thresh': cfg.get('conf_thresh', 0.0 if cfg['dataset'] == 'cityscapes' else 0.95),
        'warm_up': cfg.get('warm_up', DEFAULT_WARM_UP),
    }


def compute_total_loss(loss_x, loss_u_s1, loss_u_size, loss_u_w_fp):
    return (loss_x + 0.25 * loss_u_s1 + 0.25 * loss_u_size + 0.5 * loss_u_w_fp) / 2.0


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)
    amp = cfg.get('amp', False)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = build_scalematch_model_from_cfg(cfg)
    load_result = load_backbone_weights(model, cfg)
    if cfg.get('lock_backbone', False):
        model.lock_backbone()

    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']},
        ],
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01,
    )

    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
        logger.info('Backbone load result: %s\n', load_result)

    local_rank = int(os.environ['LOCAL_RANK'])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    model_noddp = model.module

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)

    recipe = get_scalematch_recipe(cfg)
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    best_epoch = 0
    epoch = -1
    eta_seconds = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    latest_path = os.path.join(args.save_path, 'latest.pth')
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        best_epoch = checkpoint.get('best_epoch', 0)
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        start_time = time.time()
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f} @epoch-{:}, ETA: {:.2f}M'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best, best_epoch, eta_seconds / 60
            ))

        log_avg = DictAverageMeter()
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        model.train()
        log_interval = max(len(trainloader_u) // 8, 1)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
                (img_u_w_mix, img_u_s1_mix, _, ignore_mask_mix, _, _)) in enumerate(loader):
            random_scale = random.choice(recipe['img_scales'])
            feature_scale = random.choice(recipe['feat_s_scales'] if random_scale > 1 else recipe['feat_l_scales'])

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, ignore_mask = img_u_s1.cuda(), ignore_mask.cuda()
            cutmix_box1 = cutmix_box1.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix = img_u_s1_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            iters = epoch * len(trainloader_u) + i

            cutmix_img_(img_u_s1, img_u_s1_mix, cutmix_box1)

            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp):
                    pred_u_w_mix = model_noddp(img_u_w_mix, scale_factor=None)
                    if isinstance(pred_u_w_mix, dict):
                        pred_u_w_mix = pred_u_w_mix['pred_ori']
                    conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)

                    teacher_out = model_noddp(img_u_w, scale_factor=random_scale, feature_scale=feature_scale)
                    pred_u_w = teacher_out['pred_ori'] if epoch < recipe['warm_up'] else teacher_out['pred_joint']
                    conf_u_w, mask_u_w = pred_u_w.detach().softmax(dim=1).max(dim=1)

            model.train()
            optimizer.zero_grad()

            mask_u_w_cutmixed1 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box1)
            conf_u_w_cutmixed1 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box1)
            ignore_mask_cutmixed1 = cutmix_mask(ignore_mask, ignore_mask_mix, cutmix_box1)

            with torch.cuda.amp.autocast(enabled=amp):
                num_lb = img_x.shape[0]
                student_out = model(torch.cat((img_x, img_u_w)), scale_factor=random_scale, feature_scale=feature_scale)
                pred_u_s = model(img_u_s1, scale_factor=None)
                if isinstance(pred_u_s, dict):
                    pred_u_s = pred_u_s['pred_ori']

                loss_u_s1 = criterion_u(pred_u_s, mask_u_w_cutmixed1)
                loss_u_s1 = confidence_weighted_loss(loss_u_s1, conf_u_w_cutmixed1, ignore_mask_cutmixed1, 255, conf_thresh=recipe['conf_thresh'])

                pred_x_joint = student_out['pred_joint'][:num_lb]
                pred_u_w_scale = student_out['pred_size'][num_lb:]
                pred_u_w_fp = student_out['pred_fp'][num_lb:]
                loss_x = criterion_l(pred_x_joint, mask_x)

                loss_u_size = criterion_u(pred_u_w_scale, mask_u_w)
                loss_u_size = confidence_weighted_loss(loss_u_size, conf_u_w, ignore_mask, 255, conf_thresh=recipe['conf_thresh'])

                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = confidence_weighted_loss(loss_u_w_fp, conf_u_w, ignore_mask, 255, conf_thresh=recipe['conf_thresh'])

                total_loss = compute_total_loss(loss_x, loss_u_s1, loss_u_size, loss_u_w_fp)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            valid_mask = ignore_mask != 255
            mask_ratio = ((conf_u_w >= recipe['conf_thresh']) & valid_mask).sum().float() / valid_mask.sum().clamp(min=1.0)
            log_avg.update({
                'Total_loss': total_loss,
                'Loss_x': loss_x,
                'Loss_u_s': loss_u_s1,
                'Loss_u_scale': loss_u_size,
                'Loss_w_fp_scale': loss_u_w_fp,
                'Mask_ratio': mask_ratio,
            })

            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * cfg['lr_multi']

            if (i % log_interval == 0) and (rank == 0):
                for k, v in log_avg.avgs.items():
                    writer.add_scalar('train/' + k, v.item() if torch.is_tensor(v) else v, iters)
                logger.info(f'Iters: {i}, ' + str(log_avg))
                log_avg.reset()

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        multiplier = None if cfg.get('model', 'dpt').lower() == 'upernet' else model_noddp.backbone.patch_size
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=multiplier)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % CLASSES[cfg['dataset']][i], iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if mIoU == previous_best:
            best_epoch = epoch

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'best_epoch': best_epoch,
            }
            torch.save(checkpoint, latest_path)
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

        eta_seconds = (cfg['epochs'] - (epoch + 1)) * (time.time() - start_time)


if __name__ == '__main__':
    main()
