import torch
from typing import Iterable
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import torchio as tio
from PIL import Image


class DummyToTensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)

class AddBottomLine(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        _, H, W = x.shape
        white_value = 255 if x.dtype == torch.uint8 else 1

        x[:, int(0.9*H), int(0.2*W):int(0.8*W)] = white_value
        return x
    
    
class PadTransWrapper(object):
    def __init__(self, trans, padding="zeros", img_size=32):
        super().__init__()
        self.trans = trans
        self.padding_mode = padding
        if padding != "zeros":
            self.to_pad = int(img_size // 2)
        else:
            self.to_pad = None
        self.img_size = img_size
        
    def __call__(self, x):
        # pad the image before rotate
        if self.to_pad != None:
            x = F.pad(x, (self.to_pad, self.to_pad, self.to_pad, self.to_pad), padding_mode=self.padding_mode)
        x = self.trans(x)
        # crop the image after rotate
        if self.to_pad != None:
            x = F.center_crop(x, [self.img_size, self.img_size])
        return x
    

class PadTransWrapper3D(PadTransWrapper):
    def __init__(self, trans, padding="zeros", img_size=32):
        super().__init__(trans, padding, img_size)

    def __call__(self, x):
        if self.to_pad != None:
            x = F.pad(x, (self.to_pad, self.to_pad, self.to_pad, self.to_pad, self.to_pad, self.to_pad), padding_mode=self.padding_mode)
        x = self.trans(x)
        if self.to_pad != None:
            # 3D center crop
            _, H, W, D = x.shape
            Hs = (H - self.img_size) // 2
            Ws = (W - self.img_size) // 2
            Ds = (D - self.img_size) // 2
            x = x[:, Hs:Hs+self.img_size, Ws:Ws+self.img_size, Ds:Ds+self.img_size]
        return x


class FixRotate(object):
    def __init__(self, degree, expand=False, interpolation="bilinear"):
        super().__init__()
        self.degree = degree
        self.expand = expand
        if interpolation == "nearest":
            self.interpolation = F.InterpolationMode.NEAREST
        elif interpolation == "bilinear":
            self.interpolation = F.InterpolationMode.BILINEAR
        elif interpolation == "bicubic":
            self.interpolation = F.InterpolationMode.BICUBIC

    def __call__(self, x):
        return F.rotate(x, self.degree, expand=self.expand, 
                        interpolation=self.interpolation)


class FixRotateAxis(object):
    def __init__(self, degree, axis="x", expand=False, interpolation="bilinear"):
        super().__init__()
        self.axis = axis
        self.degree = degree
        self.expand = expand
        if interpolation == "nearest":
            self.interpolation = F.InterpolationMode.NEAREST
        elif interpolation == "bilinear":
            self.interpolation = F.InterpolationMode.BILINEAR
        elif interpolation == "bicubic":
            self.interpolation = F.InterpolationMode.BICUBIC

    def __call__(self, x):
        if self.axis == 'x':
            x = F.rotate(x, self.degree, expand=self.expand,
                         interpolation=self.interpolation)
        elif self.axis == 'y':
            x = F.rotate(x.permute(0, 2, 1, 3), self.degree, expand=self.expand,
                         interpolation=self.interpolation)
            x = x.permute(0, 2, 1, 3)
        elif self.axis == 'z':
            x = F.rotate(x.permute(0, 3, 1, 2), self.degree, expand=self.expand,
                         interpolation=self.interpolation)
            x = x.permute(0, 2, 3, 1)
        else:
            raise ValueError('axis should be x, y or z')
        return x


class Padding(object):

    def __init__(self, target_size, return_array=False) -> None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.return_array = return_array

    def __process__(self, x):
        W, H = x.size
        assert H <= self.target_size[0] and W <= self.target_size[1]
        pad_h_after = int((self.target_size[0] - H) // 2)
        pad_h_before = self.target_size[0] - H - pad_h_after
        pad_w_after = int((self.target_size[1] - W) // 2)
        pad_w_before = self.target_size[1] - W - pad_w_after
        x = np.array(x)
        # deal with 3D RGB image
        if x.shape[-1] == 3:
            out = np.pad(x, [[pad_h_after, pad_h_before], [pad_w_before, pad_w_after], [0, 0]])
        else:
            out = np.pad(x, [[pad_h_after, pad_h_before], [pad_w_before, pad_w_after]])
        out = np.pad(x, [[pad_h_after, pad_h_before], [pad_w_before, pad_w_after], [0, 0]])
        if self.return_array:
            return out
        else:
            return Image.fromarray(out)

    def __call__(self, x):
        if isinstance(x, Iterable):
            out = [self.__process__(im) for im in x]
            if self.return_array:
                return np.stack(out, axis=-1)
            else:
                return out
        else:
            return self.__process__(x)



def get_mnist_transforms(args):
    if 'roteq' in args.model_type:
        image_size = (28, 28)
    else:
        image_size = (32, 32)
    transform = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    test_transform = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    if args.fix_rotate:
        test_transform.append(FixRotate(args.degree, interpolation=args.interpolation))
    if args.vflip:
        test_transform.append(transforms.RandomVerticalFlip(p=1.0))
    if args.hflip:
        test_transform.append(transforms.RandomHorizontalFlip(p=1.0))
    elif args.rotate or args.train_rotate:
        test_transform.append(transforms.RandomRotation(args.degree))
    if args.train_rotate and args.fix_rotate:
        transform.append(FixRotate(args.degree, interpolation=args.interpolation))
    elif args.train_rotate:
        transform.append(transforms.RandomRotation(args.train_degree))
    if args.test_translation:
        test_transform.append(transforms.RandomAffine(0, (args.translate_ratio, args.translate_ratio)))
    if args.translation:
        transform.append(transforms.RandomAffine(0, (args.translate_ratio, args.translate_ratio)))
    transform = transforms.Compose(transform)
    test_transform=transforms.Compose(test_transform)
    return transform, test_transform



def get_modelnet_transforms(args):
    target_size = 33 if "se3cnn" in args.model_type else 32
    if args.moco_aug:
        transform = [
            DummyToTensor(),
            tio.transforms.Resize((target_size, target_size, target_size)),
            tio.transforms.RandomAffine(scales=0.1, translation=3, degrees=0),
        ]
    else:
        transform = [
            DummyToTensor(),
            tio.transforms.Resize((target_size, target_size, target_size)),
        ]
        
    test_transform = [
        DummyToTensor(),
        tio.transforms.Resize((target_size, target_size, target_size)),
    ]
    if args.fix_rotate:
        test_transform.append(
            PadTransWrapper3D(
                FixRotateAxis(args.degree, expand=args.expand, axis=args.aug_axis, 
                              interpolation=args.interpolation),
                padding=args.padding, img_size=target_size))
    elif args.rotate or args.train_rotate:
        test_transform.append(
            PadTransWrapper3D(
                tio.transforms.RandomAffine(scales=0, degress=args.degree),
                padding=args.padding, img_size=target_size))
    if args.train_rotate and args.fix_rotate:
        transform.append(
            PadTransWrapper3D(
                FixRotateAxis(args.degree, expand=args.expand, axis=args.aug_axis, 
                              interpolation=args.interpolation),
                padding=args.padding, img_size=target_size))
    elif args.train_rotate:
        transform.append(
            PadTransWrapper3D(
                tio.transforms.RandomAffine(scales=0, degress=args.degree),
                padding=args.padding, img_size=target_size))

    if args.vflip:
        test_transform.append(tio.transforms.RandomFlip(axes=1, flip_probability=1.0))
    if args.hflip:
        test_transform.append(tio.transforms.RandomFlip(axes=2, flip_probability=1.0))
    if args.dflip:
        test_transform.append(tio.transforms.RandomFlip(axes=0, flip_probability=1.0))
    transform = transforms.Compose(transform)
    test_transform=transforms.Compose(test_transform)
    return transform, test_transform

def get_cifar10_transforms(args):
    if args.moco_aug:
        transform = [
            transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), [0.1, 2.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    else:
        transform = [
            transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    test_transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if args.res_model or args.cn_model:
        transform = [transforms.Resize((32, 32))] + transform
        test_transform = [transforms.Resize((32, 32))] + test_transform
    if args.fix_rotate:
        test_transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=32))
    elif args.rotate or args.train_rotate:
        test_transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.degree, expand=args.expand),
                padding=args.padding, img_size=32))
    if args.train_rotate and args.fix_rotate:
        transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=32))
    elif args.train_rotate:
        print(f"### training time rotation: {args.train_degree}")
        transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.train_degree, expand=args.expand), 
                padding=args.padding, img_size=32))
    if args.translation:
        transform.append(
            PadTransWrapper(
                transforms.RandomAffine(0, (args.translate_ratio, args.translate_ratio)),
                padding=args.padding, img_size=32))
    if args.test_translation:
        test_transform.append(PadTransWrapper(
                transforms.RandomAffine(0, (args.translate_ratio, args.translate_ratio)),
                padding=args.padding, img_size=32))
    if args.vflip:
        test_transform.append(transforms.RandomVerticalFlip(p=1.0))
    if args.hflip:
        test_transform.append(transforms.RandomHorizontalFlip(p=1.0))
    transform = transforms.Compose(transform)
    test_transform=transforms.Compose(test_transform)
    return transform, test_transform

def get_vhr10_transforms(args):
    if args.moco_aug:
        transform = [
            transforms.Resize((64, 64)),
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), [0.1, 2.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    else:
        transform = [
            transforms.Resize((64, 64)),
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    test_transform = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if args.fix_rotate:
        test_transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=64))
    elif args.rotate or args.train_rotate:
        test_transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.degree, expand=args.expand),
                padding=args.padding, img_size=64))
    if args.train_rotate and args.fix_rotate:
        transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=64))
    elif args.train_rotate:
        print(f"### training time rotation: {args.train_degree}")
        transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.train_degree, expand=args.expand), 
                padding=args.padding, img_size=64))
    if args.translation:
        transform.append(
            PadTransWrapper(
                transforms.RandomAffine(0, (args.translate_ratio, args.translate_ratio)),
                padding=args.padding, img_size=64))
    if args.test_translation:
        test_transform.append(
            PadTransWrapper(
                transforms.RandomAffine(0, (args.translate_ratio, args.translate_ratio)),
                padding=args.padding, img_size=64))
    if args.vflip:
        test_transform.append(transforms.RandomVerticalFlip(p=1.0))
    if args.hflip:
        test_transform.append(transforms.RandomHorizontalFlip(p=1.0))
    transform = transforms.Compose(transform)
    test_transform=transforms.Compose(test_transform)
    return transform, test_transform


def get_imagenet_transforms(args):
    if args.moco_aug:
        transform = [
            transforms.RandomResizedCrop(args.img_size, scale=(0.5, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), [0.1, 2.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:
        transform = [
            transforms.RandomResizedCrop(args.img_size, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    test_transform = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if args.res_model or args.cn_model or args.e2_model:
        transform = [transforms.Resize((args.img_size, args.img_size))] + transform
        test_transform = [transforms.Resize((args.img_size, args.img_size))] + test_transform
    if args.fix_rotate:
        test_transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=args.img_size))
    elif args.rotate or args.train_rotate:
        test_transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.degree, expand=args.expand),
                padding=args.padding, img_size=args.img_size))
    if args.train_rotate and args.fix_rotate:
        transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=args.img_size))
    elif args.train_rotate:
        print(f"### training time rotation: {args.train_degree}")
        transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.train_degree, expand=args.expand), 
                padding=args.padding, img_size=args.img_size))
    if args.translation:
        transform.append(
            PadTransWrapper(
                transforms.RandomAffine(0, (args.translate_ratio, args.translate_ratio)),
                padding=args.padding, img_size=args.img_size))
    if args.test_translation:
        args.test_translate_ratio = args.translate_ratio if args.test_translate_ratio is None else args.test_translate_ratio
        test_transform.append(PadTransWrapper(
                transforms.RandomAffine(0, (args.test_translate_ratio, args.test_translate_ratio)),
                padding=args.padding, img_size=args.img_size))
    if args.vflip:
        test_transform.append(transforms.RandomVerticalFlip(p=1.0))
    if args.hflip:
        test_transform.append(transforms.RandomHorizontalFlip(p=1.0))
    if args.random_erase:
        transform.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False))
    
    transform = transforms.Compose(transform)
    test_transform = transforms.Compose(test_transform)
    return transform, test_transform


def get_nct_crc_transforms(args):
    if args.moco_aug:
        transform = [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), [0.1, 1.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    else:
        transform = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        
    test_transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    if args.fix_rotate:
        test_transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=224))
    elif args.rotate or args.train_rotate:
        test_transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.degree, expand=args.expand),
                padding=args.padding, img_size=224))
    if args.train_rotate and args.fix_rotate:
        transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=224))
    elif args.train_rotate:
        print(f"### training time rotation: {args.train_degree}")
        transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.train_degree, expand=args.expand), 
                padding=args.padding, img_size=224))
    if args.translation:
        transform.append(
            PadTransWrapper(
                transforms.RandomAffine(0, (args.translate_ratio, args.translate_ratio)),
                padding=args.padding, img_size=224))
    if args.test_translation:
        args.test_translate_ratio = args.translate_ratio if args.test_translate_ratio is None else args.test_translate_ratio
        test_transform.append(PadTransWrapper(
                transforms.RandomAffine(0, (args.test_translate_ratio, args.test_translate_ratio)),
                padding=args.padding, img_size=224))
    if args.vflip:
        test_transform.append(transforms.RandomVerticalFlip(p=1.0))
    if args.hflip:
        test_transform.append(transforms.RandomHorizontalFlip(p=1.0))
    transform = transforms.Compose(transform)
    test_transform=transforms.Compose(test_transform)
    return transform, test_transform


def get_pcam_transforms(args):
    if args.moco_aug:
        transform = [
            transforms.Resize((96, 96)),
            transforms.RandomResizedCrop(96, scale=(0.5, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), [0.1, 1.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.7007, 0.5383, 0.6916), (0.1386, 0.1843, 0.1259)),
        ]
    else:
        transform = [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.7007, 0.5383, 0.6916), (0.1386, 0.1843, 0.1259)),
        ]
        
    test_transform = [
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.7007, 0.5383, 0.6916), (0.1386, 0.1843, 0.1259)),
    ]
    if args.fix_rotate:
        test_transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=96))
    elif args.rotate or args.train_rotate:
        test_transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.degree, expand=args.expand),
                padding=args.padding, img_size=96))
    if args.train_rotate and args.fix_rotate:
        transform.append(
            PadTransWrapper(
                FixRotate(args.degree, expand=args.expand, interpolation=args.interpolation),
                padding=args.padding, img_size=96))
    elif args.train_rotate:
        print(f"### training time rotation: {args.train_degree}")
        transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.train_degree, expand=args.expand), 
                padding=args.padding, img_size=96))
    if args.translation:
        transform.append(
            PadTransWrapper(
                transforms.RandomAffine(0, (args.translate_ratio, args.translate_ratio)),
                padding=args.padding, img_size=96))
    if args.test_translation:
        args.test_translate_ratio = args.translate_ratio if args.test_translate_ratio is None else args.test_translate_ratio
        test_transform.append(PadTransWrapper(
                transforms.RandomAffine(0, (args.test_translate_ratio, args.test_translate_ratio)),
                padding=args.padding, img_size=96))
    if args.vflip:
        test_transform.append(transforms.RandomVerticalFlip(p=1.0))
    if args.hflip:
        test_transform.append(transforms.RandomHorizontalFlip(p=1.0))
    transform = transforms.Compose(transform)
    test_transform=transforms.Compose(test_transform)
    return transform, test_transform


