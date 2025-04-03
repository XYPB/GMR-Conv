from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import torch.nn as nn
from torch import Tensor

from torchvision.transforms._presets import VideoClassification
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _KINETICS400_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

from .gmr_conv import GMR_Conv2d, GMR_Conv3d
from .gmr_resnet import _repeat

__all__ = [
    "GMRVideoResNet",
    "R3D_18_Weights",
    "r3d_18",
    "gmr_r3d_18",
]


class Conv3DSimple(nn.Conv3d):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        midplanes: Optional[int] = None,
        stride: int = 1,
        padding: int = 1,
    ) -> None:

        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride


class GMRConv3D(GMR_Conv3d):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        midplanes: Optional[int] = None,
        stride: int = 1,
        padding: int = 1,
        kernel_size: int = 3,
        num_rings: int = None,
    ) -> None:

        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=kernel_size // 2,
            num_rings=num_rings,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride




class GMRBasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        gmr_size: int = 3,
        num_rings: int = None,
    ) -> None:
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super().__init__()
        if stride > 1:
            down_pooling = nn.AvgPool3d(
                kernel_size=(stride, stride, stride), stride=(stride, stride, stride)
            )
        else:
            down_pooling = nn.Identity()
        self.conv1 = nn.Sequential(
            down_pooling,
            conv_builder(
                inplanes, planes, midplanes, 1, kernel_size=gmr_size, num_rings=num_rings
            ),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, kernel_size=gmr_size, num_rings=num_rings),
            nn.BatchNorm3d(planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GMRBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        gmr_size: int = 3,
        num_rings: int = None,
        **kwargs,
    ) -> None:

        super().__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        # Second kernel
        if stride > 1:
            down_pooling = nn.AvgPool3d(
                kernel_size=(stride, stride, stride), stride=(stride, stride, stride)
            )
        else:
            down_pooling = nn.Identity()
        self.conv2 = nn.Sequential(
            down_pooling,
            conv_builder(
                planes, planes, midplanes, 1, kernel_size=gmr_size, num_rings=num_rings
            ),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super().__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:

        super().__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(self, in_channels=3) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 1, 1),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class GMRBasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(self, in_channels=3) -> None:
        super().__init__(
            GMR_Conv3d(
                in_channels,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 1, 1),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class GMRVideoResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, GMRBasicBlock, Bottleneck]],
        conv_makers: Sequence[
            Type[Union[Conv3DSimple, GMRConv3D]]
        ],
        layers: List[int],
        stem: Callable[..., nn.Module],
        num_classes: int = 400,
        zero_init_residual: bool = False,
        in_channels: int = 3,
        gmr_conv_size: Union[int, list] = 3,
        num_rings: Union[int, list] = None,
    ) -> None:
        """Generic GMR-based resnet video generator.

        Args:
            block (Type[Union[BasicBlock, GMRBasicBlock, Bottleneck]]): resnet building block
            conv_makers (Sequence[Type[Union[Conv3DSimple, GMRConv3D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            gmr_conv_size (Union[int, list], optional): Size of GMR convolution kernels. 
                Can be an int for same size across all layers or a list for different sizes. Defaults to 3.
            num_rings (Union[int, list], optional): Number of rings for GMR convolutions.
                Can be an int for same number across all layers or a list for different numbers. Defaults to None.
        """
        super().__init__()
        self.inplanes = 64
        self.gmr_conv_size = _repeat(gmr_conv_size, 4)
        self.num_rings = _repeat(num_rings, 4)

        self.stem = stem(in_channels=in_channels)

        self.layer1 = self._make_layer(
            block,
            conv_makers[0],
            64,
            layers[0],
            stride=1,
            gmr_conv_size=self.gmr_conv_size[0],
            num_rings=self.num_rings[0],
        )
        self.layer2 = self._make_layer(
            block,
            conv_makers[1],
            128,
            layers[1],
            stride=2,
            gmr_conv_size=self.gmr_conv_size[1],
            num_rings=self.num_rings[1],
        )
        self.layer3 = self._make_layer(
            block,
            conv_makers[2],
            256,
            layers[2],
            stride=2,
            gmr_conv_size=self.gmr_conv_size[2],
            num_rings=self.num_rings[2],
        )
        self.layer4 = self._make_layer(
            block,
            conv_makers[3],
            512,
            layers[3],
            stride=2,
            gmr_conv_size=self.gmr_conv_size[3],
            num_rings=self.num_rings[3],
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, GMRBasicBlock, Bottleneck]],
        conv_builder: Type[
            Union[Conv3DSimple, GMRConv3D]
        ],
        planes: int,
        blocks: int,
        stride: int = 1,
        gmr_conv_size: int = 3,
        num_rings: int = None,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            if conv_builder == GMRConv3D:
                downsample = nn.Sequential(
                    nn.AvgPool3d(kernel_size=ds_stride, stride=ds_stride),
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=ds_stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                conv_builder,
                stride,
                downsample,
                gmr_size=gmr_conv_size,
            )
        )

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, conv_builder, gmr_size=gmr_conv_size)
            )

        return nn.Sequential(*layers)


def _video_resnet(
    block: Type[Union[BasicBlock, GMRBasicBlock, Bottleneck]],
    conv_makers: Sequence[Type[Union[Conv3DSimple]]],
    layers: List[int],
    stem: Callable[..., nn.Module],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> GMRVideoResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = GMRVideoResNet(block, conv_makers, layers, stem, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def gmr_r3d_9(*, progress: bool = True, **kwargs: Any) -> GMRVideoResNet:
    """Construct GMR 18 layer Resnet3D model.

    .. betastatus:: video module

    .. autoclass:: torchvision.models.video.R3D_18_Weights
        :members:
    """
    return _video_resnet(
        GMRBasicBlock,
        [GMRConv3D] * 4,
        [1, 1, 1, 1],
        GMRBasicStem,
        None,
        progress,
        **kwargs,
    )

def gmr_r3d_18(*, progress: bool = True, **kwargs: Any) -> GMRVideoResNet:
    """Construct GMR 18 layer Resnet3D model.

    .. betastatus:: video module

    .. autoclass:: torchvision.models.video.R3D_18_Weights
        :members:
    """
    return _video_resnet(
        GMRBasicBlock,
        [GMRConv3D] * 4,
        [2, 2, 2, 2],
        GMRBasicStem,
        None,
        progress,
        **kwargs,
    )
