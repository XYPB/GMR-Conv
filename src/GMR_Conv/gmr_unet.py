# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

from .gmr_conv import GMR_Conv2d, GMR_Conv3d
import warnings

__all__ = ["GMRUnet", "GMRunet", "GMRunet", "GMRUNet"]


class TwoConv(nn.Sequential):
    """two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        kernel_size: int = 3,
        use_gmr: bool = False,
        force_circular: bool = False,
        gaussian_smooth: bool = False,
        num_rings: int = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()

        conv_0 = Convolution(
            spatial_dims,
            in_chns,
            out_chns,
            kernel_size=kernel_size,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=int((kernel_size - 1) / 2),
        )

        conv_1 = Convolution(
            spatial_dims,
            out_chns,
            out_chns,
            kernel_size=kernel_size,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=int((kernel_size - 1) / 2),
        )

        if use_gmr:
            if spatial_dims == 2:
                conv_0.conv = GMR_Conv2d(
                    conv_0.conv.in_channels,
                    conv_0.conv.out_channels,
                    conv_0.conv.kernel_size,
                    conv_0.conv.stride,
                    conv_0.conv.padding,
                    force_circular=force_circular,
                    gaussian_mixture_ring=gaussian_smooth,
                    train_gaussian_sigma=gaussian_smooth,
                    num_rings=num_rings,
                )
                conv_1.conv = GMR_Conv2d(
                    conv_1.conv.in_channels,
                    conv_1.conv.out_channels,
                    conv_1.conv.kernel_size,
                    conv_1.conv.stride,
                    conv_1.conv.padding,
                    force_circular=force_circular,
                    gaussian_mixture_ring=gaussian_smooth,
                    train_gaussian_sigma=gaussian_smooth,
                    num_rings=num_rings,
                )
            elif spatial_dims == 3:
                conv_0.conv = GMR_Conv3d(
                    conv_0.conv.in_channels,
                    conv_0.conv.out_channels,
                    conv_0.conv.kernel_size,
                    conv_0.conv.stride,
                    conv_0.conv.padding,
                    force_circular=force_circular,
                    gaussian_mixture_ring=gaussian_smooth,
                    train_gaussian_sigma=gaussian_smooth,
                )
                conv_1.conv = GMR_Conv3d(
                    conv_1.conv.in_channels,
                    conv_1.conv.out_channels,
                    conv_1.conv.kernel_size,
                    conv_1.conv.stride,
                    conv_1.conv.padding,
                    force_circular=force_circular,
                    gaussian_mixture_ring=gaussian_smooth,
                    train_gaussian_sigma=gaussian_smooth,
                )

        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class TwoConvDecoder(nn.Sequential):
    """two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        kernel_size: int = 3,
        use_gmr: bool = False,
        force_circular: bool = False,
        gaussian_smooth: bool = False,
        num_rings: int = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()

        conv_0 = Convolution(
            spatial_dims,
            in_chns,
            out_chns,
            kernel_size=kernel_size,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=int((kernel_size - 1) / 2),
        )

        conv_1 = Convolution(
            spatial_dims,
            out_chns,
            out_chns,
            kernel_size=kernel_size,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=int((kernel_size - 1) / 2),
        )

        if use_gmr:
            if spatial_dims == 2:
                conv_0.conv = GMR_Conv2d(
                    conv_0.conv.in_channels,
                    conv_0.conv.out_channels,
                    conv_0.conv.kernel_size,
                    conv_0.conv.stride,
                    conv_0.conv.padding,
                    force_circular=force_circular,
                    gaussian_mixture_ring=gaussian_smooth,
                    train_gaussian_sigma=gaussian_smooth,
                    num_rings=num_rings,
                )
                conv_1.conv = GMR_Conv2d(
                    conv_1.conv.in_channels,
                    conv_1.conv.out_channels,
                    conv_1.conv.kernel_size,
                    conv_1.conv.stride,
                    conv_1.conv.padding,
                    force_circular=force_circular,
                    gaussian_mixture_ring=gaussian_smooth,
                    train_gaussian_sigma=gaussian_smooth,
                    num_rings=num_rings,
                )
            elif spatial_dims == 3:
                conv_0.conv = GMR_Conv3d(
                    conv_0.conv.in_channels,
                    conv_0.conv.out_channels,
                    conv_0.conv.kernel_size,
                    conv_0.conv.stride,
                    conv_0.conv.padding,
                    force_circular=force_circular,
                    gaussian_mixture_ring=gaussian_smooth,
                    train_gaussian_sigma=gaussian_smooth,
                )
                conv_1.conv = GMR_Conv3d(
                    conv_1.conv.in_channels,
                    conv_1.conv.out_channels,
                    conv_1.conv.kernel_size,
                    conv_1.conv.stride,
                    conv_1.conv.padding,
                    force_circular=force_circular,
                    gaussian_mixture_ring=gaussian_smooth,
                    train_gaussian_sigma=gaussian_smooth,
                )

        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        kernel_size: int = 3,
        use_gmr: bool = False,
        force_circular: bool = False,
        gaussian_smooth: bool = False,
        num_rings: int = None,
        pooling_type: str = "MAX",
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()
        max_pooling = Pool[pooling_type, spatial_dims](kernel_size=2)
        convs = TwoConv(
            spatial_dims,
            in_chns,
            out_chns,
            act,
            norm,
            bias,
            dropout,
            kernel_size=kernel_size,
            use_gmr=use_gmr,
            force_circular=force_circular,
            gaussian_smooth=gaussian_smooth,
            num_rings=num_rings,
        )
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        is_pad: bool = True,
        kernel_size: int = 3,
        use_gmr: bool = False,
        force_circular: bool = False,
        gaussian_smooth: bool = False,
        num_rings: int = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the encoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.
            is_pad: whether to pad upsampling features to fit features from encoder. Defaults to True.

        """
        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConvDecoder(
            spatial_dims,
            cat_chns + up_chns,
            out_chns,
            act,
            norm,
            bias,
            dropout,
            kernel_size=kernel_size,
            use_gmr=use_gmr,
            force_circular=force_circular,
            gaussian_smooth=gaussian_smooth,
            num_rings=num_rings,
        )
        self.is_pad = is_pad

        # if upsample == "nontrainable" and halves:
        #     print('Use simple upsample with Interpolate Mode', interp_mode)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            if self.is_pad:
                # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
                dimensions = len(x.shape) - 2
                sp = [0] * (dimensions * 2)
                for i in range(dimensions):
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(
                torch.cat([x_e, x_0], dim=1)
            )  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x


class GMRUNet(nn.Module):
    @deprecated_arg(
        name="dimensions",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = (
            "LeakyReLU",
            {"negative_slope": 0.1, "inplace": True},
        ),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        interp_mode: str = "linear",
        dimensions: Optional[int] = None,
        kernels: Sequence[int] = (9, 9, 5, 5, 3),
        use_gmr: bool = True,
        force_circular: bool = False,
        gaussian_smooth: bool = False,
        layers: int = 4,
        num_rings: int = None,
        pooling_type: str = "MAX",
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            jz729: new args:
            use_gmr: whether to use GMR conv.
            force_circular: whether to force circular.
            gaussian_smooth: whether to perform gaussian smoothing on the kernel.
            layers: number of downsamplings.
            kernels: kernel size of conv in each layer.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = GMRUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = GMRUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = GMRUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)
        if num_rings is None:
            num_rings = [None] * 5
        print(f"GMRUNet features: {fea[:layers+1]+fea[5:]}.")
        print(
            "Migrate from MONAI GMRUNet with changeable kernel size and GMR compatibility."
        )

        self.layers = layers
        if layers + 1 != len(kernels):
            warnings.warn(
                f"`layers+1 ({layers+1}) != len(kernels) ({kernels})`, needs adjustment."
            )

        self.conv_0 = TwoConv(
            spatial_dims,
            in_channels,
            features[0],
            act,
            norm,
            bias,
            dropout,
            use_gmr=use_gmr,
            force_circular=force_circular,
            gaussian_smooth=gaussian_smooth,
            kernel_size=kernels[0],
            num_rings=num_rings[0],
        )
        self.down_1 = Down(
            spatial_dims,
            fea[0],
            fea[1],
            act,
            norm,
            bias,
            dropout,
            use_gmr=use_gmr,
            force_circular=force_circular,
            gaussian_smooth=gaussian_smooth,
            kernel_size=kernels[1],
            num_rings=num_rings[1],
            pooling_type=pooling_type,
        )

        if layers == 4:
            self.down_2 = Down(
                spatial_dims,
                fea[1],
                fea[2],
                act,
                norm,
                bias,
                dropout,
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                kernel_size=kernels[2],
                num_rings=num_rings[2],
                pooling_type=pooling_type,
            )
            self.down_3 = Down(
                spatial_dims,
                fea[2],
                fea[3],
                act,
                norm,
                bias,
                dropout,
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                kernel_size=kernels[3],
                num_rings=num_rings[3],
                pooling_type=pooling_type,
            )
            self.down_4 = Down(
                spatial_dims,
                fea[3],
                fea[4],
                act,
                norm,
                bias,
                dropout,
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                kernel_size=kernels[4],
                num_rings=num_rings[4],
                pooling_type=pooling_type,
            )

            self.upcat_4 = UpCat(
                spatial_dims,
                fea[4],
                fea[3],
                fea[3],
                act,
                norm,
                bias,
                dropout,
                upsample,
                interp_mode=interp_mode,
                kernel_size=kernels[3],
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                num_rings=num_rings[3],
            )
            self.upcat_3 = UpCat(
                spatial_dims,
                fea[3],
                fea[2],
                fea[2],
                act,
                norm,
                bias,
                dropout,
                upsample,
                interp_mode=interp_mode,
                kernel_size=kernels[2],
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                num_rings=num_rings[2],
            )
            self.upcat_2 = UpCat(
                spatial_dims,
                fea[2],
                fea[1],
                fea[1],
                act,
                norm,
                bias,
                dropout,
                upsample,
                interp_mode=interp_mode,
                kernel_size=kernels[1],
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                num_rings=num_rings[1],
            )
        if layers == 3:
            self.down_2 = Down(
                spatial_dims,
                fea[1],
                fea[2],
                act,
                norm,
                bias,
                dropout,
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                kernel_size=kernels[2],
                num_rings=num_rings[2],
                pooling_type=pooling_type,
            )
            self.down_3 = Down(
                spatial_dims,
                fea[2],
                fea[3],
                act,
                norm,
                bias,
                dropout,
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                kernel_size=kernels[3],
                num_rings=num_rings[3],
                pooling_type=pooling_type,
            )

            self.upcat_3 = UpCat(
                spatial_dims,
                fea[3],
                fea[2],
                fea[2],
                act,
                norm,
                bias,
                dropout,
                upsample,
                interp_mode=interp_mode,
                kernel_size=kernels[2],
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                num_rings=num_rings[2],
            )
            self.upcat_2 = UpCat(
                spatial_dims,
                fea[2],
                fea[1],
                fea[1],
                act,
                norm,
                bias,
                dropout,
                upsample,
                interp_mode=interp_mode,
                kernel_size=kernels[1],
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                num_rings=num_rings[1],
            )
        if layers == 2:
            self.down_2 = Down(
                spatial_dims,
                fea[1],
                fea[2],
                act,
                norm,
                bias,
                dropout,
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                kernel_size=kernels[2],
                num_rings=num_rings[2],
                pooling_type=pooling_type,
            )

            self.upcat_2 = UpCat(
                spatial_dims,
                fea[2],
                fea[1],
                fea[1],
                act,
                norm,
                bias,
                dropout,
                upsample,
                interp_mode=interp_mode,
                kernel_size=kernels[1],
                use_gmr=use_gmr,
                force_circular=force_circular,
                gaussian_smooth=gaussian_smooth,
                num_rings=num_rings[1],
            )

        self.upcat_1 = UpCat(
            spatial_dims,
            fea[1],
            fea[0],
            fea[5],
            act,
            norm,
            bias,
            dropout,
            upsample,
            interp_mode=interp_mode,
            halves=False,
            kernel_size=kernels[0],
            use_gmr=use_gmr,
            force_circular=force_circular,
            gaussian_smooth=gaussian_smooth,
            num_rings=num_rings[0],
        )

        self.final_conv = Conv["conv", spatial_dims](
            fea[5], out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `spatial_dims`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        x0 = self.conv_0(x)

        if self.layers == 4:
            x1 = self.down_1(x0)
            x2 = self.down_2(x1)
            x3 = self.down_3(x2)
            x4 = self.down_4(x3)

            u4 = self.upcat_4(x4, x3)
            u3 = self.upcat_3(u4, x2)
            u2 = self.upcat_2(u3, x1)
            u1 = self.upcat_1(u2, x0)
        elif self.layers == 3:
            x1 = self.down_1(x0)
            x2 = self.down_2(x1)
            x3 = self.down_3(x2)

            u3 = self.upcat_3(x3, x2)
            u2 = self.upcat_2(u3, x1)
            u1 = self.upcat_1(u2, x0)
        elif self.layers == 2:
            x1 = self.down_1(x0)
            x2 = self.down_2(x1)

            u2 = self.upcat_2(x2, x1)
            u1 = self.upcat_1(u2, x0)
        elif self.layers == 1:
            x1 = self.down_1(x0)

            u1 = self.upcat_1(x1, x0)
        else:
            raise ValueError(
                f"Number of layers `layers` ({self.layers}) must be 1 to 4."
            )

        logits = self.final_conv(u1)
        return logits


GMRUnet = GMRunet = GMRunet = GMRUNet
