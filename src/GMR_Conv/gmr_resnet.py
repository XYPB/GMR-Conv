import math
from itertools import repeat
from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .gmr_conv import GMR_Conv2d


def _repeat(x, t):
    if not isinstance(x, list):
        return list(repeat(x, t))
    elif len(x) == 1:
        return list(repeat(x[0], t))
    else:
        return x


def conv1x1(
    in_planes: int, out_planes: int, stride: int = 1, bias: bool = False
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def GMR_convkxk(
    k: int,
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    bias: bool = False,
    num_rings: int = None,
    sigma_no_weight_decay: bool = False,
) -> GMR_Conv2d:
    """3x3 RI convolution with padding"""
    return GMR_Conv2d(
        in_planes,
        out_planes,
        kernel_size=k,
        stride=stride,
        padding=int(k // 2),
        groups=groups,
        bias=bias,
        dilation=dilation,
        num_rings=num_rings,
        sigma_no_weight_decay=sigma_no_weight_decay,
    )


class GMRBasicBlock(nn.Module):
    """
    Avoid the issue due to RIconv with stride differ to 1
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        gmr_conv_size: int = 3,
        gmr_groups: int = 1,
        num_rings: int = None,
        sigma_no_weight_decay: bool = False,
        **kwargs: Any,
    ) -> None:
        super(GMRBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.stride = stride
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conv1 = GMR_convkxk(
            gmr_conv_size,
            inplanes,
            planes,
            groups=gmr_groups,
            num_rings=num_rings,
            sigma_no_weight_decay=sigma_no_weight_decay,
        )
        if self.stride != 1:
            self.conv1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=self.stride, stride=self.stride),
                conv1,
            )
        else:
            self.conv1 = conv1
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = GMR_convkxk(
            gmr_conv_size,
            planes,
            planes,
            groups=gmr_groups,
            num_rings=num_rings,
            sigma_no_weight_decay=sigma_no_weight_decay,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GMRBottleneck(nn.Module):
    """
    Symmetric Residual Bottleneck Block
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        gmr_conv_size: int = 3,
        gmr_groups: int = 1,
        num_rings: int = None,
        sigma_no_weight_decay: bool = False,
        **kwargs: Any,
    ) -> None:
        super(GMRBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        gmr_groups = width if gmr_groups != 1 else 1
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # conv1 and conv3 is the same in all case
        conv2 = GMR_convkxk(
            gmr_conv_size,
            width,
            width,
            stride=1,
            groups=gmr_groups,
            num_rings=num_rings,
            sigma_no_weight_decay=sigma_no_weight_decay,
        )
        if self.stride != 1:
            self.conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=self.stride, stride=self.stride), conv2
            )
        else:
            self.conv2 = conv2
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GMR_ResNet(nn.Module):
    """
    GMR_ResNet implements a ResNet architecture with Gaussian Mixture Ring convolutional layers.
    
    Args:
        block (Type[Union[GMRBasicBlock, GMRBottleneck]]): The block type to use throughout the network 
            (either GMRBasicBlock or GMRBottleneck)
        layers (List[int]): Number of blocks in each of the four layers of the network
        num_classes (int): Number of output classes. Default: 1000
        zero_init_residual (bool): If True, zero-initialize the last BN in each residual branch. Default: False
        groups (int): Number of groups for the GroupedConv. Default: 1
        width_per_group (int): Width of each group in the GroupedConv. Default: 64
        replace_stride_with_dilation (Optional[List[bool]]): Replace stride with dilation in each layer. Default: None
        norm_layer (Optional[Callable[..., nn.Module]]): Normalization layer. Default: None (BatchNorm2d)
        gmr_conv_size (Union[int, list]): Size of the GMR convolutional kernel, can be a list of length 4 
            to specify different sizes for each layer. Default: 3
        inplanes (int): Number of base channels in the network. Default: 64
        layer_stride (Type[Union[int, List[int]]]): Stride for each layer. Default: [1, 2, 2, 2]
        num_rings (Union[int, list]): Number of rings in the GMR kernel, can be a list to specify 
            different values for each layer. Default: None (automatically determined)
        skip_first_maxpool (bool): If True, skip the first maxpool layer, useful for small images. Default: False
        sigma_no_weight_decay (bool): If True, exclude sigma parameters from weight decay. Default: False
        in_channels (int): Number of input channels. Default: 3
    """

    def __init__(
        self,
        block: Type[Union[GMRBasicBlock, GMRBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        gmr_conv_size: Union[int, list] = 3,
        inplanes: int = 64,
        layer_stride: Type[Union[int, List[int]]] = [1, 2, 2, 2],
        num_rings: Union[int, list] = None,
        skip_first_maxpool: bool = False,
        sigma_no_weight_decay: bool = False,
        in_channels: int = 3,
        **kwargs,
    ) -> None:
        super(GMR_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.gmr_conv_size = _repeat(gmr_conv_size, 4)
        self.num_rings = _repeat(num_rings, 4)
        if isinstance(layer_stride, float):
            layer_stride = [layer_stride for _ in range(4)]
        else:
            layer_stride = layer_stride

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = GMR_Conv2d(
            in_channels, self.inplanes, kernel_size=5, stride=1, padding=2, bias=False,
            sigma_no_weight_decay=sigma_no_weight_decay,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if skip_first_maxpool:
            self.first_avgpool = nn.Identity()
        else:
            self.first_avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(
            block,
            inplanes,
            layers[0],
            stride=layer_stride[0],
            num_rings=self.num_rings[0],
            gmr_conv_size=self.gmr_conv_size[0],
            sigma_no_weight_decay=sigma_no_weight_decay,
        )
        self.layer2 = self._make_layer(
            block,
            2 * inplanes,
            layers[1],
            stride=layer_stride[1],
            dilate=replace_stride_with_dilation[0],
            num_rings=self.num_rings[1],
            gmr_conv_size=self.gmr_conv_size[1],
            sigma_no_weight_decay=sigma_no_weight_decay,
        )
        self.layer3 = self._make_layer(
            block,
            4 * inplanes,
            layers[2],
            stride=layer_stride[2],
            dilate=replace_stride_with_dilation[1],
            num_rings=self.num_rings[2],
            gmr_conv_size=self.gmr_conv_size[2],
            sigma_no_weight_decay=sigma_no_weight_decay,
        )
        self.layer4 = self._make_layer(
            block,
            8 * inplanes,
            layers[3],
            stride=layer_stride[3],
            dilate=replace_stride_with_dilation[2],
            num_rings=self.num_rings[3],
            gmr_conv_size=self.gmr_conv_size[3],
            sigma_no_weight_decay=sigma_no_weight_decay,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(8 * inplanes * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, GMR_Conv2d):
                _, fan_out = nn.init._calculate_fan_in_and_fan_out(
                    torch.zeros(m.weight_matrix_shape)
                )
                gain = nn.init.calculate_gain("relu", 0)
                std = gain / math.sqrt(fan_out)
                nn.init.normal_(m.weight, 0, std)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, GMRBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, GMRBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[GMRBasicBlock, GMRBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        gmr_conv_size: int = 3,
        num_rings: int = None,
        sigma_no_weight_decay: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )

        layers = []
        gmr_groups = 1
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                gmr_conv_size=gmr_conv_size,
                gmr_groups=gmr_groups,
                num_rings=num_rings,
                sigma_no_weight_decay=sigma_no_weight_decay,
            )
        )
        self.inplanes = planes * block.expansion
        gmr_groups = 1
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    gmr_conv_size=gmr_conv_size,
                    gmr_groups=gmr_groups,
                    num_rings=num_rings,
                    sigma_no_weight_decay=sigma_no_weight_decay,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.first_avgpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[GMRBasicBlock, GMRBottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> GMR_ResNet:
    model = GMR_ResNet(block, layers, **kwargs)
    return model


def gmr_resnet18(**kwargs: Any) -> GMR_ResNet:
    r"""GMR_ResNet-18 model from
    'GMR-Conv: An Efficient Rotation and Reflection Equivariant Convolution Kernel Using Gaussian Mixture Rings'
    """
    return _resnet("GMR_resnet18", GMRBasicBlock, [2, 2, 2, 2], **kwargs)


def gmr_resnet34(**kwargs: Any) -> GMR_ResNet:
    r"""GMR_ResNet-34 model from
    'GMR-Conv: An Efficient Rotation and Reflection Equivariant Convolution Kernel Using Gaussian Mixture Rings'
    """
    return _resnet("GMR_resnet34", GMRBasicBlock, [3, 4, 6, 3], **kwargs)


def gmr_resnet50(**kwargs: Any) -> GMR_ResNet:
    r"""GMR_ResNet-50 model from
    'GMR-Conv: An Efficient Rotation and Reflection Equivariant Convolution Kernel Using Gaussian Mixture Rings'
    """
    return _resnet("GMR_resnet50", GMRBottleneck, [3, 4, 6, 3], **kwargs)
