import json
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from contrast import resnet
from contrast.data import get_loader
from contrast.logger import setup_logger
from contrast.lr_scheduler import get_scheduler
from contrast.option import parse_option
from contrast.util import AverageMeter, accuracy, reduce_tensor

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def build_model(args, num_class):
    # create model
    model = resnet.__dict__[args.arch](low_dim=num_class, head_type='reduce').cuda()

    # set requires_grad of parameters except last fc layer to False
    for name, p in model.named_parameters():
        if 'fc' not in name:
            p.requires_grad = False

    optimizer = torch.optim.SGD(model.fc.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    return model, optimizer


def load_pretrained(model, pretrained_model):
    ckpt = torch.load(pretrained_model, map_location='cpu')
    model_dict = model.state_dict()

    base_fix = False
    for key in ckpt['model'].keys():
        if key.startswith('module.base.'):
            base_fix = True
            break

    if base_fix:
        state_dict = {k.replace("module.base.", "module."): v
                      for k, v in ckpt['model'].items()
                      if k.startswith('module.base.')}
        logger.info(f"==> load checkpoint from Module.Base")
    else:
        state_dict = {k.replace("module.encoder.", "module."): v
                      for k, v in ckpt['model'].items()
                      if k.startswith('module.encoder.')}
        logger.info(f"==> load checkpoint from Module.Encoder")

    state_dict = {k: v for k, v in state_dict.items()
                  if k in model_dict and v.size() == model_dict[k].size()}

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    logger.info(f"==> loaded checkpoint '{pretrained_model}' (epoch {ckpt['epoch']})")


def load_checkpoint(args, model, optimizer, scheduler):
    logger.info("=> loading checkpoint '{args.resume'")

    checkpoint = torch.load(args.resume, map_location='cpu')

    global best_acc1
    best_acc1 = checkpoint['best_acc1']
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if args.amp_opt_level != "O0" and checkpoint['args'].amp_opt_level != "O0":
        amp.load_state_dict(checkpoint['amp'])

    logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")


def save_checkpoint(args, epoch, model, test_acc, optimizer, scheduler):
    state = {
        'args': args,
        'epoch': epoch,
        'model': model.state_dict(),
        'best_acc1': test_acc,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    if args.amp_opt_level != "O0":
        state['amp'] = amp.state_dict()
    torch.save(state, os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth'))
    torch.save(state, os.path.join(args.output_dir, f'current.pth'))


def main(args):
    global best_acc1

    args.batch_size = args.total_batch_size // dist.get_world_size()
    train_loader = get_loader(args.aug, args, prefix='train')
    val_loader = get_loader('val', args, prefix='val')
    logger.info(f"length of training dataset: {len(train_loader.dataset)}")

    model, optimizer = build_model(args, num_class=len(train_loader.dataset.classes))
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    # load pre-trained model
    load_pretrained(model, args.pretrained_model)

    # optionally resume from a checkpoint
    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, "current.pth")
        if os.path.exists(resume_file):
            logger.info(f'auto resume from {resume_file}')
            args.resume = resume_file
        else:
            logger.info(f'no checkpoint found in {args.output_dir}, ignoring auto resume')
    if args.resume:
        assert os.path.isfile(args.resume), f"no checkpoint found at '{args.resume}'"
        load_checkpoint(args, model, optimizer, scheduler)

    if args.eval:
        logger.info("==> testing...")
        validate(val_loader, model, args)
        return

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        tic = time.time()
        train(epoch, train_loader, model, optimizer, scheduler, args)
        logger.info(f'epoch {epoch}, total time {time.time() - tic:.2f}')

        logger.info("==> testing...")
        test_acc, test_acc5, test_loss = validate(val_loader, model, args)
        if summary_writer is not None:
            summary_writer.add_scalar('test_acc', test_acc, epoch)
            summary_writer.add_scalar('test_acc5', test_acc5, epoch)
            summary_writer.add_scalar('test_loss', test_loss, epoch)

        # save model
        if dist.get_rank() == 0 and epoch % args.save_freq == 0:
            logger.info('==> Saving...')
            save_checkpoint(args, epoch, model, test_acc, optimizer, scheduler)


def train(epoch, train_loader, model, optimizer, scheduler, args):
    """
    one epoch training
    """

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (x, _, y) in enumerate(train_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        output = model(x)
        loss = F.cross_entropy(output, y)

        # backward
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        # update meters
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        loss_meter.update(loss.item(), x.size(0))
        acc1_meter.update(acc1[0], x.size(0))
        acc5_meter.update(acc5[0], x.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0:
            logger.info(
                f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Lr {optimizer.param_groups[0]["lr"]:.3f} \t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})')

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def validate(val_loader, model, args):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (x, _, y) in enumerate(val_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            # compute output
            output = model(x)
            loss = F.cross_entropy(output, y)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, y, topk=(1, 5))

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter.update(loss.item(), x.size(0))
            acc1_meter.update(acc1[0], x.size(0))
            acc5_meter.update(acc5[0], x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})')

        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    opt = parse_option(stage='linear')

    if opt.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True
    best_acc1 = 0

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="contrast")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, "w") as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(vars(opt))

    main(opt)
