from __future__ import print_function
import os
from glob import glob
import builtins
import shutil
import time
from pytz import timezone
from datetime import datetime
import json
import copy
import wandb

import numpy as np
import torch
from torch.backends import cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
import gc

from opt import get_opt
from GMR_Conv.transforms import (
    get_mnist_transforms,
    get_cifar10_transforms,
    get_vhr10_transforms,
    get_imagenet_transforms,
    get_nct_crc_transforms,
    get_pcam_transforms,
    get_modelnet_transforms,
)
import torchvision.models.resnet as resnet
import torchvision.models.video.resnet as resnet_3d
import GMR_Conv.gmr_resnet as gmr_resnet
import GMR_Conv.gmr_resnet_3d as gmr_resnet_3d
from GMR_Conv.gmr_conv import GMR_Conv2d
from datasets.mnist import MNIST
from datasets.vhr10 import VHR10
from datasets.mtarsi import MTARSI
from datasets.nct_crc import NCT_CRC
from datasets.imagenet_val import ImageNetVal
from datasets.pcam import PatchCamelyon
from datasets.modelnet import MyModelNet
from utils import (
    manual_seed,
    one_hot,
    log_train_val,
    acc_at_topk,
)


# @profile
def train(
    args,
    model,
    device,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    epoch,
    log_dir,
    logger=None,
):
    if log_dir:
        train_log_dir = os.path.join(log_dir, "vis_train")
        os.makedirs(train_log_dir, exist_ok=True)
    model.train()
    step_losses = []
    outputs = []
    targets = []
    step_grad_norm = []
    st = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if len(target.shape) == 2:
            target = target.squeeze(1)
        if args.dev:
            print(data.shape, target.shape)
            print(torch.unique(target.detach().cpu()))
        dtype = torch.bfloat16 if args.bf16 else None
        with torch.autocast("cuda", enabled=args.amp, dtype=dtype):
            output = model(data)
            if args.multi_label:
                loss = F.binary_cross_entropy_with_logits(
                    output, target.to(torch.float32)
                )
            else:
                loss = F.cross_entropy(
                    output, target, label_smoothing=args.label_smoothing
                )
            if args.dev:
                print(data.shape, output.shape, data[:, 0, :10, 0], output)
        # autocast should only wrap the forward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if args.warm_up >= 0:
            scheduler.step()
        optimizer.zero_grad()
        if args.bf16:
            output = output.detach().float()
        else:
            output = output.detach()

        # logging
        step_losses.append(loss.item())
        outputs.append(output.cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        if args.log_grad_norm:
            grad_norm = 0
            for p in model.parameters():
                para_norm = p.grad.data.norm(2)
                grad_norm += para_norm.item() ** 2
            step_grad_norm.append(np.sqrt(grad_norm))
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAvg. Loss: {np.mean(step_losses):.6f}\tTime: {time.time()-st:.2f}"
            )
            if args.dry_run:
                return step_losses, 0, step_grad_norm
        if logger:
            logger.log(
                {
                    "train/loss": loss.item(),
                    "train/avg_loss": np.mean(step_losses),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )
    et = time.time()
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0).squeeze()
    if args.multi_label:
        preds = (outputs > 0.5).astype(int)
        acc_top5 = 0.0
        auc = 100 * roc_auc_score(targets, outputs, multi_class="ovr")
        f1 = 100 * f1_score(targets, preds, average="macro")
        acc = 100 * accuracy_score(targets.flatten(), preds.flatten())
    else:
        preds = np.argmax(outputs, axis=1).squeeze()
        acc_top5 = 100 * acc_at_topk(targets, outputs, 5)
        auc = 100 * roc_auc_score(
            one_hot(targets, num_classes=outputs.shape[1]), outputs, multi_class="ovr"
        )
        f1 = 100 * f1_score(targets, preds, average="macro")
        acc = 100 * accuracy_score(targets, preds)
    duration = et - st if et - st < 60 else (et - st) / 60
    duration = f"{duration:.2f}min" if et - st > 60 else f"{duration:.2f}s"
    print(
        f"Train set average Acc@1: {acc:.2f}%,\t Acc@5: {acc_top5:.2f}%,\tAUC: {auc:.2f}%,\tF1-score: {f1:.2f}%,\tTime: {duration}"
    )
    if logger:
        logger.log(
            {
                "train/epoch_acc": acc,
                "train/epoch_acc_top5": acc_top5,
                "train/epoch_auc": auc,
                "train/epoch_f1": f1,
            }
        )
    del targets, outputs
    return step_losses, acc, step_grad_norm


def test(
    args,
    model,
    device,
    test_loader,
    epoch,
    verbose=True,
    confusion_mat=False,
    logger=None,
):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    st = time.time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if len(target.shape) == 2:
                target = target.squeeze(1)
            dtype = torch.bfloat16 if args.bf16 else None
            with torch.autocast("cuda", enabled=args.amp, dtype=dtype):
                if args.dev and batch_idx == 0:
                    print(data.shape, target.shape)
                output = model(data.to(torch.float32))
                if args.multi_label:
                    loss = F.binary_cross_entropy_with_logits(
                        output, target.to(torch.float32), reduction="sum"
                    )
                else:
                    loss = F.cross_entropy(output, target, reduction="sum")
            test_loss += loss.item()  # sum up batch loss
            if args.bf16:
                output = output.detach().float()
            else:
                output = output.detach()
            outputs.append(output.cpu().numpy())
            targets.append(target.detach().cpu().numpy())
    et = time.time()

    test_loss /= len(test_loader.dataset)
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0).squeeze()
    if args.multi_label:
        preds = (outputs > 0.5).astype(int)
        acc_top5 = 0.0
        auc = 100 * roc_auc_score(targets, outputs, multi_class="ovr")
        f1 = 100 * f1_score(targets, preds, average="macro")
        ba = 0.0
        acc = 100 * accuracy_score(targets.flatten(), preds.flatten())
    else:
        preds = np.argmax(outputs, axis=1).squeeze()
        acc_top5 = 100 * acc_at_topk(targets, outputs, 5)
        auc = 100 * roc_auc_score(
            one_hot(targets, num_classes=outputs.shape[1]), outputs, multi_class="ovr"
        )
        f1 = 100 * f1_score(targets, preds, average="macro")
        ba = 100 * balanced_accuracy_score(targets, preds)
        acc = 100 * accuracy_score(targets, preds)
    correct = np.sum(targets == preds)
    if logger:
        logger.log(
            {
                "test/epoch_acc": acc,
                "test/epoch_acc_top5": acc_top5,
                "test/epoch_auc": auc,
                "test/epoch_f1": f1,
                "test/epoch_ba": ba,
            }
        )

    if verbose:
        duration = et - st if et - st < 60 else (et - st) / 60
        duration = f"{duration:.2f}min" if et - st > 60 else f"{duration:.2f}s"
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%), Acc@5: {acc_top5:.2f}%, AUC: {auc:.2f}%, BA: {ba:.2f}%, F1-score: {f1:.2f}% Time: {duration}\n"
        )
        if confusion_mat:
            print(f"Confusion Matrix:\n{confusion_matrix(targets, preds)}\n")
    return test_loss, acc


def eval_rot(model, test_loader, device, args, verbose=False, eval_3d=False):
    overall_acc = []
    print("## Evaluate on rotation:")
    degrees = range(0, 361, 30) if eval_3d else range(0, 361, 10)
    axes = ["x", "y", "z"] if eval_3d else [""]
    for ax in axes:
        for deg in degrees:
            if verbose:
                print(f"## Evaluate on rotation: {deg} and axis: {ax}")
            args.degree = deg
            args.fix_rotate = True
            if args.cifar10 or args.cifar100:
                _, test_transform = get_cifar10_transforms(args)
            elif args.vhr10 or args.mtarsi:
                _, test_transform = get_vhr10_transforms(args)
            elif args.imagenet:
                # go with full image size
                _, test_transform = get_imagenet_transforms(args)
            elif args.nct_crc:
                _, test_transform = get_nct_crc_transforms(args)
            elif args.pcam:
                _, test_transform = get_pcam_transforms(args)
            elif args.modelnet10 or args.modelnet40:
                _, test_transform = get_modelnet_transforms(args)
            else:
                _, test_transform = get_mnist_transforms(args)
            # hack to test rotation
            test_loader.dataset.transform = test_transform
            test_loss, test_acc = test(
                args, model, device, test_loader, epoch=0, verbose=verbose
            )
            overall_acc.append(round(test_acc, 2))
    print(f"## Overall Acc. for all degrees: {overall_acc}")
    print(
        f"## Overall Avg. Acc.: {np.round(np.mean(overall_acc), 2)}({np.round(np.std(overall_acc), 2)})"
    )
    return np.mean(overall_acc)


def eval_flip(model, test_loader, device, args, verbose=False, eval_3d=False):
    overall_acc = []
    flipping = ["", "v", "h", "d"] if eval_3d else ["", "v", "h"]
    print("## Evaluate on flipping:")
    for flip in flipping:
        if verbose:
            print(f"## Evaluate on flip: {flip}")
        if "v" in flip:
            args.vflip = True
        if "h" in flip:
            args.hflip = True
        if "d" in flip:
            args.dflip = True
        if args.cifar10 or args.cifar100:
            _, test_transform = get_cifar10_transforms(args)
        elif args.vhr10 or args.mtarsi:
            _, test_transform = get_vhr10_transforms(args)
        elif args.imagenet:
            # go with full image size
            _, test_transform = get_imagenet_transforms(args)
        elif args.nct_crc:
            _, test_transform = get_nct_crc_transforms(args)
        elif args.pcam:
            _, test_transform = get_pcam_transforms(args)
        elif args.modelnet10 or args.modelnet40:
            _, test_transform = get_modelnet_transforms(args)
        else:
            _, test_transform = get_mnist_transforms(args)
        # hack to test rotation
        test_loader.dataset.transform = test_transform
        test_loss, test_acc = test(
            args, model, device, test_loader, epoch=0, verbose=verbose
        )
        overall_acc.append(round(test_acc, 2))
        # Reset
        args.vflip = False
        args.hflip = False
        args.dflip = False
    print(f"## Overall Acc. for all flipping: {overall_acc}")
    print(
        f"## Overall Avg. Acc.: {np.round(np.mean(overall_acc[1:]), 2)}({np.round(np.std(overall_acc), 2)})"
    )
    return np.mean(overall_acc)


# @profile
def main_worker(rank, world_size, args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.ddp:
        print(f"Use GPU:{rank} for training")
        setup(rank, world_size, args)

        # suppress printing if not master
        if rank != 0 and not args.dev:

            def print_pass(*args):
                pass

            builtins.print = print_pass

    manual_seed(args.seed)
    logger = None
    if args.save_model or args.log:
        est = timezone("US/Eastern")
        dt = est.localize(datetime.now())
        dt_str = dt.strftime("%Y-%m-%d-%H-%M-%S")
        if args.cifar10:
            task_name = "cifar10"
        elif args.cifar100:
            task_name = "cifar100"
        elif args.imagenet:
            task_name = "imagenet"
        elif args.vhr10:
            task_name = "vhr10"
        elif args.mtarsi:
            task_name = "mtarsi"
        elif args.nct_crc:
            task_name = "nct_crc"
        elif args.pcam:
            task_name = "pcam"
        elif args.modelnet10:
            task_name = "modelnet10"
        elif args.modelnet40:
            task_name = "modelnet40"
        else:
            task_name = "mnist"
        log_dir = os.path.join(
            args.base_log_dir,
            f"{task_name}_{dt_str}_{args.model_type}_{args.exp}_train_logs",
        )
        config_dir = os.path.join(log_dir, "config")
        if args.dev or args.dry_run:
            logger = None
        else:
            if rank == 0:
                logger = wandb.init(
                    project=f"RotInvCls_{task_name}",
                    config=args.__dict__,
                    name=log_dir.split("/")[-1].replace("_train_logs", ""),
                )
    else:
        log_dir = None

    if use_cuda:
        if args.ddp:
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.cudnn:
        cudnn.benchmark = True

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.batch_size}
    if use_cuda:
        num_cores = len(os.sched_getaffinity(0))
        if args.num_workers > num_cores:
            args.num_workers = num_cores
        print(f"### Use {args.num_workers} cores for training...")
        cuda_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": False,
            "shuffle": True,
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    if "ric" in args.model_type:
        train_kwargs["drop_last"] = True
        test_kwargs["drop_last"] = True

    ####### DATASET #######
    in_channels = 3
    if args.cifar10:
        transform, test_transform = get_cifar10_transforms(args)
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )
        n_classes = 10
        input_shape = (in_channels, 32, 32)
    elif args.cifar100:
        transform, test_transform = get_cifar10_transforms(args)
        train_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=test_transform
        )
        n_classes = 100
        input_shape = (in_channels, 32, 32)
    elif args.vhr10:
        transform, test_transform = get_vhr10_transforms(args)
        train_dataset = VHR10(
            root="./data/NWPU_VHR-10_dataset/instance_image_set",
            train=True,
            split_file="./data/NWPU_VHR-10_dataset/VHR_split.json",
            transform=transform,
        )
        test_dataset = VHR10(
            root="./data/NWPU_VHR-10_dataset/instance_image_set",
            train=False,
            split_file="./data/NWPU_VHR-10_dataset/VHR_split.json",
            transform=test_transform,
        )
        n_classes = 10
        input_shape = (in_channels, 64, 64)
    elif args.mtarsi:
        transform, test_transform = get_vhr10_transforms(args)
        train_dataset = MTARSI(
            train=True,
            split_file="./data/MTARSI_1/MTARSI_1_split.json",
            transform=transform,
        )
        test_dataset = MTARSI(
            train=False,
            split_file="./data/MTARSI_1/MTARSI_1_split.json",
            transform=test_transform,
        )
        n_classes = 20
        input_shape = (in_channels, 64, 64)
    elif args.imagenet:
        transform, test_transform = get_imagenet_transforms(args)
        train_dataset = torchvision.datasets.ImageFolder(
            root="./data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
            transform=transform,
        )
        test_dataset = ImageNetVal(
            root="./data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
            transform=test_transform,
            class_to_idx=train_dataset.class_to_idx,
        )
        n_classes = 1000
        input_shape = (in_channels, args.img_size, args.img_size)
    elif args.nct_crc:
        transform, test_transform = get_nct_crc_transforms(args)
        train_dataset = NCT_CRC("data/NCT-CRC-HE-100K", transform)
        test_dataset = NCT_CRC("data/CRC-VAL-HE-7K", test_transform)
        n_classes = 9
        input_shape = (in_channels, 224, 224)
    elif args.pcam:
        transform, test_transform = get_pcam_transforms(args)
        train_dataset = PatchCamelyon(
            "data/PatchCamelyon/pcam/training_split.h5",
            "data/PatchCamelyon/Labels/Labels/camelyonpatch_level_2_split_train_y.h5",
            transform,
        )
        test_dataset = PatchCamelyon(
            "data/PatchCamelyon/pcam/test_split.h5",
            "data/PatchCamelyon/Labels/Labels/camelyonpatch_level_2_split_test_y.h5",
            test_transform,
        )
        n_classes = 2
        input_shape = (in_channels, 96, 96)
    elif args.modelnet10:
        in_channels = 1
        transform, test_transform = get_modelnet_transforms(args)
        train_dataset = MyModelNet(name="10", train=True, transform=transform)
        test_dataset = MyModelNet(name="10", train=False, transform=test_transform)
        n_classes = 10
        input_shape = (in_channels, 33, 33, 33)
    elif args.modelnet40:
        in_channels = 1
        transform, test_transform = get_modelnet_transforms(args)
        train_dataset = MyModelNet(name="40", train=True, transform=transform)
        test_dataset = MyModelNet(name="40", train=False, transform=test_transform)
        n_classes = 40
        input_shape = (in_channels, 33, 33, 33)
    else:
        transform, test_transform = get_mnist_transforms(args)
        train_dataset = MNIST("../data", train=True, download=True, transform=transform)
        test_dataset = MNIST("../data", train=False, transform=test_transform)
        in_channels = 1
        n_classes = 10
        input_shape = (in_channels, 32, 32)

    ####### MODEL #######
    if args.modelnet10 or args.modelnet40:
        if "gmr" in args.model_type:
            if args.gmr_conv_size_list != None:
                gmr_conv_size = args.gmr_conv_size_list
            else:
                gmr_conv_size = args.gmr_conv_size
            model = getattr(gmr_resnet_3d, args.model_type)(
                num_classes=n_classes,
                in_channels=in_channels,
                in_planes=args.res_inplanes,
                gmr_conv_size=gmr_conv_size,
                num_rings=args.num_rings,
            )
        else:
            model = getattr(resnet_3d, "r3d_18")(
                num_classes=n_classes,
                in_channels=in_channels,
                in_planes=args.res_inplanes,
            )
    else:
        if args.gmr_conv_size_list != None:
            gmr_conv_size = args.gmr_conv_size_list
        else:
            gmr_conv_size = args.gmr_conv_size
        if "gmr" in args.model_type:
            model = getattr(gmr_resnet, args.model_type)(
                num_classes=n_classes,
                inplanes=args.res_inplanes,
                gmr_conv_size=gmr_conv_size,
                num_rings=args.num_rings,
                in_channels=in_channels,
            )
        else:
            # You may need to modify the resnet model to accept 1 channel input
            model = getattr(resnet, args.model_type)(
                num_classes=n_classes,
                inplanes=args.res_inplanes,
                in_channels=in_channels,
            )
        # remove the first downsampling to ensure last stage input size
        if args.res_keep_conv1:
            if "gmr" in args.model_type:
                model.conv1 = GMR_Conv2d(
                    in_channels,
                    args.res_inplanes,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    bias=False,
                )
            else:
                model.conv1 = nn.Conv2d(
                    in_channels,
                    args.res_inplanes,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    bias=False,
                )

    if args.dev:
        print(model)

    elif args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        train_kwargs.pop("shuffle")
        train_kwargs["sampler"] = train_sampler

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    model = model.to(device)

    #### Scaler
    scaler = GradScaler(enabled=args.scaler)

    param_cnt = sum(
        [param.numel() for param in model.parameters() if param.requires_grad]
    )
    print(f"Correct Total params: {param_cnt}")
    
    # Make the sigma values not to be regularized by weight decay
    params = [
        param for name, param in model.named_parameters() if "sigmas" not in name
    ]
    sigma_params = [
        param for name, param in model.named_parameters() if "sigmas" in name
    ]
    param_group = [{"params": sigma_params, "weight_decay": 0}, {"params": params}]

    if args.adam:
        optimizer = optim.Adam(param_group, lr=args.lr, weight_decay=args.weight_decay)
    elif args.sgd:
        optimizer = optim.SGD(
            param_group,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.adamW:
        optimizer = optim.AdamW(param_group, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adadelta(param_group, lr=args.lr)

    cur_ep = 1
    if args.resume:
        args.log = True
        args.save_model = True
        print(f"### resume experiment logged under {args.resume}...")
        log_dir = args.resume
        config_dir = os.path.join(log_dir, "config")
        ckpt_dir = os.path.join(log_dir, "ckpt")
        assert os.path.exists(log_dir)
        assert os.path.exists(config_dir)
        assert os.path.exists(ckpt_dir)
        args.load_model = log_dir
    elif args.log or args.save_model:
        try:
            if args.ddp:
                if rank == 0:
                    os.makedirs(log_dir, exist_ok=False)
                dist.barrier()
            else:
                os.makedirs(log_dir, exist_ok=False)
        except FileExistsError:
            cnt = glob(log_dir)
            log_dir = log_dir.replace("_train_logs", f"_train_logs_{len(cnt)}")
            os.makedirs(log_dir, exist_ok=False)
        print(f"### experiment logged under {log_dir}")
        os.makedirs(config_dir, exist_ok=True)
        arg_dict = vars(args)
        json.dump(arg_dict, open(os.path.join(config_dir, "train_config.json"), "w"))

    ckpt = None
    if args.load_model:
        assert os.path.exists(args.load_model)
        print(f"### load model logged under {args.load_model}...")
        model_ckpt_dir = os.path.join(args.load_model, "ckpt")
        ckpts = sorted(glob(os.path.join(model_ckpt_dir, "*.ckpt")))
        ckpt_path = ckpts[-1]
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["state_dict"]
        cur_ep = ckpt["cur_ep"]
        print(f"### load model {ckpt_path} at epoch {cur_ep}...")
        target_state_dict = {}
        for k, param in state_dict.items():
            k = k.replace("module.", "")
            target_state_dict[k] = param
        model.load_state_dict(target_state_dict)

    if args.cos:
        if args.warm_up > 0:
            first_cycle_steps = args.epochs * len(train_loader)
            warmup_steps = args.warm_up * len(train_loader)
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=first_cycle_steps,
                cycle_mult=1.0,
                max_lr=args.lr,
                min_lr=args.min_lr,
                warmup_steps=warmup_steps,
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=cur_ep - 2
            )
    elif args.step:
        scheduler = StepLR(
            optimizer, step_size=1, gamma=args.gamma, last_epoch=cur_ep - 2
        )
    elif args.multi_step:
        scheduler = MultiStepLR(optimizer, [50, 75], gamma=0.1, last_epoch=cur_ep - 2)
    else:
        scheduler = LambdaLR(optimizer, lambda x: x, last_epoch=cur_ep - 2)

    if ckpt and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f"### load optimizer and scheduler from {ckpt_path}...")

    if args.ddp:
        model = DDP(model, device_ids=[rank])

    total_train_loss = []
    total_train_acc = []
    total_grad_norm = []
    total_test_loss = []
    total_test_acc = []
    total_eval_rot_acc = []
    total_eval_flip_acc = []
    best_acc = -1
    cur_ep -= 1 if args.eval_only and cur_ep == args.epochs + 1 else 0
    for epoch in range(cur_ep, args.epochs + 1):

        if not args.eval_only:
            step_losses, train_acc, grad_norm = train(
                args,
                model,
                device,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                epoch,
                log_dir,
                logger,
            )
            confusion_mat = (
                args.utp
                or args.embed
                or args.balance_embed
                or args.embed_density
                or args.roi_screen_embed
                or args.pcam
            )
            test_loss, test_acc = test(
                args,
                model,
                device,
                test_loader,
                epoch=epoch,
                confusion_mat=confusion_mat,
                logger=logger,
            )
            steps = len(step_losses)
            total_train_loss += step_losses
            total_train_acc += [train_acc for _ in range(steps)]
            if args.log_grad_norm:
                total_grad_norm += grad_norm
            total_test_loss += [test_loss for _ in range(steps)]
            total_test_acc += [test_acc for _ in range(steps)]
            if epoch == 1 and rank == 0:
                print(torch.cuda.memory_summary(device=device, abbreviated=False))

            if args.warm_up < 0:
                scheduler.step()

            if args.log:
                total_grad_norm = total_grad_norm if args.log_grad_norm else None
                if logger is not None:
                    log_train_val(
                        total_train_loss,
                        total_test_loss,
                        total_train_acc,
                        total_test_acc,
                        total_grad_norm,
                        log_dir,
                    )

            if args.save_model:
                ckpt_dir = os.path.join(log_dir, "ckpt")
                os.makedirs(ckpt_dir, exist_ok=True)
                if args.save_rec and ((epoch + 1) % args.save_interval) == 0:
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "cur_ep": epoch + 1,
                    }
                    torch.save(
                        ckpt,
                        os.path.join(
                            ckpt_dir, f"{args.model_type}_ep{epoch:0>4d}.ckpt"
                        ),
                    )
                elif args.save_best:
                    ckpt_dest = os.path.join(ckpt_dir, f"{args.model_type}_last.ckpt")
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "cur_ep": epoch + 1,
                    }
                    torch.save(ckpt, ckpt_dest)
                    if test_acc > best_acc:
                        best_dist = os.path.join(
                            ckpt_dir, f"{args.model_type}_best.ckpt"
                        )
                        print(f"### Update best weight with test auc: {test_acc:.4f}")
                        shutil.copy(ckpt_dest, best_dist)
                        best_acc = test_acc
                else:
                    ckpt_dest = os.path.join(ckpt_dir, f"{args.model_type}_last.ckpt")
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "cur_ep": epoch + 1,
                    }
                    torch.save(ckpt, ckpt_dest)

        if args.ddp:
            dist.barrier()

        gc.collect()

        verbose = epoch == args.epochs
        eval_3d = (args.med_mnist != None and "3d" in args.med_mnist) or args.modelnet10 or args.modelnet40
        if args.eval_rot and (epoch % args.eval_interval == 0 or epoch == args.epochs):
            # don't change the original args
            rot_acc = eval_rot(
                model,
                test_loader,
                device,
                copy.deepcopy(args),
                verbose=verbose,
                eval_3d=eval_3d,
            )
            total_eval_rot_acc.append(rot_acc)
            if logger:
                logger.log({"test/eval_rot_acc": rot_acc})

        if args.eval_flip and (epoch % args.eval_interval == 0 or epoch == args.epochs):
            # don't change the original args
            flip_acc = eval_flip(
                model,
                test_loader,
                device,
                copy.deepcopy(args),
                verbose=verbose,
                eval_3d=eval_3d,
            )
            total_eval_flip_acc.append(flip_acc)
            if logger:
                logger.log({"test/eval_flip_acc": flip_acc})

    if rank == 0 and args.log:
        with open(os.path.join(log_dir, "logs.json"), "w") as f:
            serialize = lambda l: [float(x) for x in l]
            logs = {
                "train_loss": serialize(total_train_loss),
                "test_loss": serialize(total_test_loss),
                "train_acc": serialize(total_train_acc),
                "test_acc": serialize(total_test_acc),
                "eval_rot_acc": serialize(total_eval_rot_acc),
                "eval_flip_acc": serialize(total_eval_flip_acc),
            }
            json.dump(logs, f)
        print(f"### experiment logged under {log_dir}")

    if logger:
        logger.finish()

    if args.ddp:
        cleanup()


def setup(rank, world_size, args):
    # initialize the process group
    dist.init_process_group(
        "nccl", init_method=args.dist_url, rank=rank, world_size=world_size
    )


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    args = get_opt()
    if args.ddp:
        world_size = args.world_size
        torch.multiprocessing.spawn(
            main_worker,
            args=(
                world_size,
                args,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        main_worker(0, 1, args)
