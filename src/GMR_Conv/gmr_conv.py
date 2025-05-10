import math
import collections
from itertools import repeat
from typing import List, Union, Tuple, Optional

from scipy import ndimage
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


def clip_module_index_mat(model, eps=0.1):
    for module in model.modules():
        if isinstance(module, GMR_Conv2d):
            module.clip_index_mat(eps)


class _GMR_ConvNd(nn.Module):
    r"""
    Official implementation of Symmetric Rotation Invariant Conv2d in PyTorch.
    Apply symmetric rotation invariant convolution over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        transposed (bool, optional): If ``True``, use a transposed convolution operator. Default: ``False``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode (str, optional): Accepted values `zeros`, `reflect`, `replicate`, `circular`. Default: `zeros`
        num_rings (int, optional): Number of bands in the GMR kernel. If None, calculate based on kernel size. Default: None
        train_index_mat (bool, optional): If ``True``, train the index matrix. Default: ``True``
        inference_accelerate (bool, optional): If ``True``, accelerate inference by pre-computing the weight matrix. Default: ``True``
        force_circular (bool, optional): If ``True``, force the kernel to be circular by ignoring corner part. Default: ``True``
        gaussian_mixture_ring (bool, optional): If ``True``, diffuse the weight index mat with gaussian mixture of rings. Default: ``True``
        train_gaussian_sigma (bool, optional): If ``True``, train the sigma of gaussian distribution. Default: ``True``
        gaussian_sigma_scale (float, optional): Scale of sigma of gaussian distribution. Default: ``2.355``
        clip_sigma (bool, optional): If ``True``, clip the sigma of gaussian distribution. Default: ``True``
        dw_fwd (bool, optional): If ``True``, use depthwise-like efficient forward. Default: ``True``
        device (torch.device, optional): Device to use. Default: ``None``
        dtype (torch.dtype, optional): Data type to use. Default: ``None``

    Attributes:
        weight (torch.Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{ri\_k})`
        bias (torch.Tensor):   the learnable bias of the module of shape
            :math:`(\text{out\_channels})`
        weight_index_mat (torch.Tensor): the learnable index matrix of the module of shape
            :math:`(\text{in\_channels}, \text{out\_channels}, \text{ri\_k}, \text{ri\_k})`
        infer_weight_matrix (torch.Tensor): the pre-computed weight matrix of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{H}, \text{W})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...] = 1,
        padding: Union[str, Tuple[int, ...]] = 0,
        dilation: Tuple[int, ...] = 1,
        transposed: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        num_rings: int = None,
        train_index_mat: bool = True,
        inference_accelerate: bool = True,
        force_circular: bool = True,
        gaussian_mixture_ring: bool = True,
        train_gaussian_sigma: bool = True,
        sigma_no_weight_decay: bool = False,
        gaussian_sigma_scale: float = 2.355,
        clip_sigma: bool = True,
        dw_fwd: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # TODO: this part goes to individual conv
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings
                    )
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )
        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes, padding_mode
                )
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.groups = groups
        self.padding_mode = padding_mode
        self.output_padding = self.padding
        self.train_index_mat = train_index_mat
        self.inference_accelerate = inference_accelerate
        self.force_circular = force_circular
        self.gaussian_mixture_ring = gaussian_mixture_ring
        self.train_gaussian_sigma = train_gaussian_sigma
        self.dw_fwd = dw_fwd
        self.device = device
        self.dtype = dtype
        self.clip_sigma = clip_sigma

        self.num_rings = (kernel_size[0] // 2) + 1
        if not self.force_circular:
            self.num_rings += 1
        self.num_rings = self.num_rings if num_rings == None else num_rings

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if transposed:
            self.weight = nn.Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, self.num_rings),
                    **factory_kwargs,
                )
            )
            self.weight_matrix_shape = (
                in_channels,
                out_channels // groups,
                *kernel_size,
            )
            weight_index_mat = self._make_weight_index_mat(1, 1, factory_kwargs)
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, self.num_rings),
                    **factory_kwargs,
                )
            )
            self.weight_matrix_shape = (
                out_channels,
                in_channels // groups,
                *kernel_size,
            )
            weight_index_mat = self._make_weight_index_mat(1, 1, factory_kwargs)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # use to clip the weight index mat
        self.register_buffer("default_weight_index_mat", weight_index_mat)
        if self.train_index_mat:
            self.weight_index_mat = nn.Parameter(weight_index_mat)
            self.weight_index_mat.requires_grad = True
        else:
            self.register_buffer("weight_index_mat", weight_index_mat)

        if self.gaussian_mixture_ring:
            mus = torch.arange(self.num_rings, **factory_kwargs)
            sigmas = torch.empty(self.num_rings, **factory_kwargs)
            # init sigma to be FWHM of bands width
            band_width = self.kernel_size[0] / (2 * (self.num_rings - 1))
            sigmas[:] = band_width / gaussian_sigma_scale
            # Register parameters/buffers
            self.register_buffer("mus", mus)
            if self.train_gaussian_sigma:
                self.log_sigmas = nn.Parameter(torch.log(sigmas), requires_grad=True)
                # Add to the list of parameters not to apply weight decay
                # when initialize the optim
                if sigma_no_weight_decay:
                    self.log_sigmas._no_weight_decay = True
            else:
                self.register_buffer("log_sigmas", torch.log(sigmas))
        else:
            self.register_buffer("sigmas", None)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        # manually init weight to avoid problem
        fan, _ = init._calculate_fan_in_and_fan_out(
            torch.zeros(self.weight_matrix_shape)
        )
        gain = init.calculate_gain("leaky_relu", math.sqrt(5))
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_GMR_ConvNd, self).__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"

    def clip_index_mat(self, eps=0.1):
        if not self.train_index_mat:
            return
        # only clip the index mat after each param update
        _max = self.default_weight_index_mat + eps
        _min = self.default_weight_index_mat - eps
        self.weight_index_mat.data = torch.clamp(
            self.weight_index_mat, min=_min, max=_max
        )

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if self.inference_accelerate and not mode:
            self.infer_weight_matrix = self._make_weight_matrix(self.weight)
        else:
            self.infer_weight_matrix = None
        return super().train(mode)

    def _get_index_matrix(self):
        # Note that einsum is generally faster than batch matrix multiplication
        if self.gaussian_mixture_ring:
            # diffuse the weight index mat with gaussian distribution
            # This step is necessary to make the sigma value differentiable
            # Clamp sigma to avoid nan
            # Empirically best choice for sigma range is [1e-2, 2*num_rings]
            mus = self.mus[None, :]
            sigmas = torch.exp(self.log_sigmas)[:None]
            if self.clip_sigma:
                sigmas = torch.clamp(sigmas, min=1e-2, max=2 * self.num_rings)
            loc = torch.arange(
                self.num_rings,
                device=self.log_sigmas.device,
                dtype=self.log_sigmas.dtype,
            )[:, None].repeat(1, self.num_rings)
            prob = 1 / (sigmas * 2.5066) * torch.exp(-0.5 * ((loc - mus) / sigmas) ** 2)
            prob = prob.permute(1, 0)
            # No need to re-parameterization trick since no sampling involved
            # need to consider wether to normalize the probability
            # Normalization will results in difference between each band
            # and there is no need since the weight matrix will learn the scale by itself
            # if self.normalize_gaussian_prob:
            #     prob = prob / prob.sum(dim=1, keepdim=True)
            # print(prob)
            weight_index_mat = prob @ self.weight_index_mat[0, 0][None, None]
            # weight_index_mat = torch.einsum('lk,ijkw->ijlw', prob, self.weight_index_mat)

        else:
            weight_index_mat = self.weight_index_mat
        return weight_index_mat

    def _make_weight_matrix(self, weight):
        weight_index_mat = self._get_index_matrix()

        weight = torch.einsum("ijkw,ijk->ijw", weight_index_mat, weight)
        weight = weight.reshape(self.weight_matrix_shape)
        return weight

    def _make_weight_index_mat(
        self, index_mat_C_in, index_mat_C_out, factory_kwargs
    ): ...

    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor: ...


class GMR_Conv1d(_GMR_ConvNd):
    """
    GMR_Conv1d is a 1-dimensional convolutional layer that incorporates Gaussian Mixture Rings (GMR)
    for enhanced feature extraction. This layer extends the basic convolutional layer by adding
    several parameters and methods to support GMR.
    Args:
        in_channels (int): Number of channels in the input signal.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (_size_1_t): Size of the convolving kernel.
        stride (_size_1_t, optional): Stride of the convolution. Default: 1.
        padding (Union[str, _size_1_t], optional): Padding added to both sides of the input. Default: 0.
        dilation (_size_1_t, optional): Spacing between kernel elements. Default: 1.
        transposed (bool, optional): If True, use transposed convolution. Default: False.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True.
        padding_mode (str, optional): 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'.
        num_rings (int, optional): Number of rings in the Gaussian Mixture. Default: None.
        train_index_mat (bool, optional): If True, the index matrix is trainable. Default: True.
        inference_accelerate (bool, optional): If True, use accelerated inference. Default: True.
        force_circular (bool, optional): If True, force circular convolution. Default: True.
        gaussian_mixture_ring (bool, optional): If True, use Gaussian Mixture Rings. Default: True.
        train_gaussian_sigma (bool, optional): If True, the Gaussian sigma is trainable. Default: True.
        sigma_no_weight_decay (bool, optional): If True, sigma has no weight decay. Default: False.
        gaussian_sigma_scale (float, optional): Scale factor for Gaussian sigma. Default: 2.355.
        dw_fwd (bool, optional): If True, use depthwise-like efficient forward. Default: True.
        device (optional): Device to use for computation. Default: None.
        dtype (optional): Data type to use for computation. Default: None.
        **kwargs: Additional arguments.
    Methods:
        _make_weight_index_mat(index_mat_C_in, index_mat_C_out, factory_kwargs):
            Creates the weight index matrix for the convolution.
        _conv_forward(input, weight, bias):
            Performs the forward pass of the convolution.
        forward(input):
            Defines the computation performed at every call.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        transposed: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        num_rings: int = None,
        train_index_mat: bool = True,
        inference_accelerate: bool = True,
        force_circular: bool = True,
        gaussian_mixture_ring: bool = True,
        train_gaussian_sigma: bool = True,
        sigma_no_weight_decay: bool = False,
        gaussian_sigma_scale: float = 2.355,
        dw_fwd: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        dilation = _single(dilation)
        padding = padding if isinstance(padding, str) else _single(padding)
        super(GMR_Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            groups,
            bias,
            padding_mode,
            num_rings,
            train_index_mat,
            inference_accelerate,
            force_circular,
            gaussian_mixture_ring,
            train_gaussian_sigma,
            sigma_no_weight_decay,
            gaussian_sigma_scale,
            dw_fwd,
            **factory_kwargs,
        )

    def _make_weight_index_mat(self, index_mat_C_in, index_mat_C_out, factory_kwargs):
        weight_index_mats = []
        _, _, H = self.weight_matrix_shape
        D = np.ones((H))
        D[H // 2] = 0
        D = ndimage.distance_transform_edt(D)
        max_dist = (H // 2) + 0.5 if self.force_circular else D.max()
        num_levels = self.num_rings + 1 if self.force_circular else self.num_rings
        levels = np.linspace(D.min(), max_dist, num=num_levels)
        for i in range(num_levels):
            if i == num_levels - 1:
                if self.force_circular:
                    continue
                idx = (D == levels[i]).astype(int)
            else:
                idx = ((D >= levels[i]) & (D < levels[i + 1])).astype(int)
            level_mat = torch.tensor(idx, **factory_kwargs)[None, None, :, :]
            weight_index_mats.append(level_mat.reshape(-1))
        weight_index_mats = torch.stack(weight_index_mats, dim=0)
        weight_index_mats = weight_index_mats.to(torch.float32)
        weight_index_mats = weight_index_mats.expand(
            (index_mat_C_in, index_mat_C_out, -1, -1)
        )
        return weight_index_mats

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input):
        if not self.training and self.inference_accelerate:
            weight_matrix = self.infer_weight_matrix
            weight_matrix = weight_matrix.to(device=input.device, dtype=input.dtype)
        else:
            weight_matrix = self._make_weight_matrix(self.weight)
        return self._conv_forward(input, weight_matrix, self.bias)


class GMR_Conv2d(_GMR_ConvNd):
    """
    GMR_Conv2d is a custom 2D convolutional layer that extends the _GMR_ConvNd class.
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (_size_2_t): Size of the convolving kernel.
        stride (_size_2_t, optional): Stride of the convolution. Default is 1.
        padding (Union[str, _size_2_t], optional): Padding added to all four sides of the input. Default is 0.
        dilation (_size_2_t, optional): Spacing between kernel elements. Default is 1.
        transposed (bool, optional): If True, use transposed convolution. Default is False.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is True.
        padding_mode (str, optional): 'zeros', 'reflect', 'replicate' or 'circular'. Default is 'zeros'.
        num_rings (int, optional): Number of rings for Gaussian Mixture Ring. Default is None.
        train_index_mat (bool, optional): If True, train the index matrix. Default is True.
        inference_accelerate (bool, optional): If True, accelerate inference. Default is True.
        force_circular (bool, optional): If True, force circular convolution. Default is True.
        gaussian_mixture_ring (bool, optional): If True, use Gaussian Mixture Ring. Default is True.
        train_gaussian_sigma (bool, optional): If True, train the Gaussian sigma. Default is True.
        sigma_no_weight_decay (bool, optional): If True, do not apply weight decay to sigma. Default is False.
        gaussian_sigma_scale (float, optional): Scale for Gaussian sigma. Default is 2.355.
        dw_fwd (bool, optional): If True, use depthwise-like efficient forward. Default is True.
        device (optional): Device on which to place the tensor. Default is None.
        dtype (optional): Data type of the tensor. Default is None.
        **kwargs: Additional keyword arguments.
    Methods:
        _make_weight_index_mat(index_mat_C_in, index_mat_C_out, factory_kwargs):
            Creates the weight index matrix.
        _conv_forward(input, weight, bias):
            Performs the convolution operation.
        _costume_forward_dw(input, return_all=False):
            Performs the depthwise-like efficient forward operation.
        forward(input):
            Defines the computation performed at every call.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        transposed: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        num_rings: int = None,
        train_index_mat: bool = True,
        inference_accelerate: bool = True,
        force_circular: bool = True,
        gaussian_mixture_ring: bool = True,
        train_gaussian_sigma: bool = True,
        sigma_no_weight_decay: bool = False,
        gaussian_sigma_scale: float = 2.355,
        dw_fwd: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        padding = padding if isinstance(padding, str) else _pair(padding)
        super(GMR_Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            groups,
            bias,
            padding_mode,
            num_rings,
            train_index_mat,
            inference_accelerate,
            force_circular,
            gaussian_mixture_ring,
            train_gaussian_sigma,
            sigma_no_weight_decay,
            gaussian_sigma_scale,
            dw_fwd,
            **factory_kwargs,
        )

    def _make_weight_index_mat(self, index_mat_C_in, index_mat_C_out, factory_kwargs):
        weight_index_mats = []
        _, _, H, W = self.weight_matrix_shape
        D = np.ones((H, W))
        D[(H // 2, W // 2)] = 0
        D = ndimage.distance_transform_edt(D)
        max_dist = (H // 2) + 0.5 if self.force_circular else D.max()
        num_levels = self.num_rings + 1 if self.force_circular else self.num_rings
        levels = np.linspace(D.min(), max_dist, num=num_levels)
        for i in range(num_levels):
            if i == num_levels - 1:
                if self.force_circular:
                    continue
                idx = (D == levels[i]).astype(int)
            else:
                idx = ((D >= levels[i]) & (D < levels[i + 1])).astype(int)
            level_mat = torch.tensor(idx, **factory_kwargs)[None, None, :, :]
            weight_index_mats.append(level_mat.reshape(-1))
        weight_index_mats = torch.stack(weight_index_mats, dim=0)
        weight_index_mats = weight_index_mats.to(torch.float32)
        weight_index_mats = weight_index_mats.expand(
            (index_mat_C_in, index_mat_C_out, -1, -1)
        )
        return weight_index_mats

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def _costume_forward_dw(self, input, return_all=False):
        weight_index_mat = self._get_index_matrix()[0, 0]  # b, k^2

        weight_index_mat = weight_index_mat.reshape(
            -1, self.kernel_size[0], self.kernel_size[0]
        )[:, None]  # b, 1, k, k
        B, Cin, H, W = input.shape
        input = input.reshape(B * Cin, 1, H, W)
        output1 = F.conv2d(
            input, weight_index_mat, None, 1, self.padding, self.dilation, 1
        )  # (B*Cin, b, H, W)
        weight = self.weight.reshape(
            self.out_channels, self.in_channels * self.num_rings, 1, 1
        )  # Cout, Cin*b, 1, 1
        H, W = output1.shape[2:]
        output2 = F.conv2d(
            output1.reshape(B, Cin * self.num_rings, H, W),
            weight,
            self.bias,
            self.stride,
            0,
            1,
            1,
        )  # (B, Cout, H, W)
        if return_all:
            return output2, output1
        else:
            return output2

    def forward(self, input):
        if self.dw_fwd:
            return self._costume_forward_dw(input)
        if not self.training and self.inference_accelerate:
            weight_matrix = self.infer_weight_matrix
            weight_matrix = weight_matrix.to(device=input.device, dtype=input.dtype)
        else:
            weight_matrix = self._make_weight_matrix(self.weight)
        return self._conv_forward(input, weight_matrix, self.bias)


class GMR_Conv3d(_GMR_ConvNd):
    """
    GMR_Conv3d is a custom 3D convolutional layer that extends the _GMR_ConvNd class.
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (_size_2_t): Size of the convolving kernel.
        stride (_size_2_t, optional): Stride of the convolution. Default is 1.
        padding (Union[str, _size_2_t], optional): Padding added to all four sides of the input. Default is 0.
        dilation (_size_2_t, optional): Spacing between kernel elements. Default is 1.
        transposed (bool, optional): If True, use transposed convolution. Default is False.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is True.
        padding_mode (str, optional): 'zeros', 'reflect', 'replicate' or 'circular'. Default is 'zeros'.
        num_rings (int, optional): Number of rings for Gaussian Mixture Ring. Default is None.
        train_index_mat (bool, optional): If True, train the index matrix. Default is True.
        inference_accelerate (bool, optional): If True, accelerate inference. Default is True.
        force_circular (bool, optional): If True, force circular convolution. Default is True.
        gaussian_mixture_ring (bool, optional): If True, use Gaussian Mixture Ring. Default is True.
        train_gaussian_sigma (bool, optional): If True, train the Gaussian sigma. Default is True.
        sigma_no_weight_decay (bool, optional): If True, do not apply weight decay to sigma. Default is False.
        gaussian_sigma_scale (float, optional): Scale for Gaussian sigma. Default is 2.355.
        dw_fwd (bool, optional): If True, use depthwise-like efficient forward. Default is True.
        device (optional): Device on which to place the tensor. Default is None.
        dtype (optional): Data type of the tensor. Default is None.
        **kwargs: Additional keyword arguments.
    Methods:
        _make_weight_index_mat(index_mat_C_in, index_mat_C_out, factory_kwargs):
            Creates the weight index matrix.
        _conv_forward(input, weight, bias):
            Performs the convolution operation.
        _costume_forward_dw(input, return_all=False):
            Performs the depthwise-like efficient forward operation.
        forward(input):
            Defines the computation performed at every call.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        transposed: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        num_rings: int = None,
        train_index_mat: bool = True,
        inference_accelerate: bool = True,
        force_circular: bool = True,
        gaussian_mixture_ring: bool = True,
        sigma_no_weight_decay: bool = False,
        train_gaussian_sigma: bool = True,
        gaussian_sigma_scale: float = 2.355,
        dw_fwd: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        dilation = _triple(dilation)
        padding = padding if isinstance(padding, str) else _triple(padding)
        super(GMR_Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            groups,
            bias,
            padding_mode,
            num_rings,
            train_index_mat,
            inference_accelerate,
            force_circular,
            gaussian_mixture_ring,
            train_gaussian_sigma,
            sigma_no_weight_decay,
            gaussian_sigma_scale,
            dw_fwd,
            **factory_kwargs,
        )

    def _make_weight_index_mat(self, index_mat_C_in, index_mat_C_out, factory_kwargs):
        weight_index_mats = []
        _, _, H, W, L = self.weight_matrix_shape
        D = np.ones((H, W, L))
        D[(H // 2, W // 2, L // 2)] = 0
        D = ndimage.distance_transform_edt(D)
        max_dist = (H // 2) + 0.5 if self.force_circular else D.max()
        num_levels = self.num_rings + 1 if self.force_circular else self.num_rings
        levels = np.linspace(D.min(), max_dist, num=num_levels)
        for i in range(num_levels):
            if i == num_levels - 1:
                if self.force_circular:
                    continue
                idx = (D == levels[i]).astype(int)
            else:
                idx = ((D >= levels[i]) & (D < levels[i + 1])).astype(int)
            level_mat = torch.tensor(idx, **factory_kwargs)[None, None, :, :, :]
            weight_index_mats.append(level_mat.reshape(-1))
        weight_index_mats = torch.stack(weight_index_mats, dim=0)
        weight_index_mats = weight_index_mats.to(torch.float32)
        weight_index_mats = weight_index_mats.expand(
            (index_mat_C_in, index_mat_C_out, -1, -1)
        )
        return weight_index_mats

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input):
        if not self.training and self.inference_accelerate:
            weight_matrix = self.infer_weight_matrix
            weight_matrix = weight_matrix.to(device=input.device, dtype=input.dtype)
        else:
            weight_matrix = self._make_weight_matrix(self.weight)
        return self._conv_forward(input, weight_matrix, self.bias)


class GMR_ConvTranspose2d(GMR_Conv2d):
    """
    GMR_ConvTranspose2d is a transposed 2D convolutional layer that extends the _GMR_ConvNd class.
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True.
        padding_mode (str, optional): Only 'zeros' is supported. Default: 'zeros'.
        num_rings (int, optional): Number of rings for the convolution. Default: None.
        train_index_mat (bool, optional): If True, trains the index matrix. Default: False.
        inference_accelerate (bool, optional): If True, accelerates inference. Default: True.
        force_circular (bool, optional): If True, forces circular convolution. Default: True.
        dw_fwd (bool, optional): Depthwise-like efficient forward flag. Default: True.
        device (torch.device, optional): The desired device of the parameters. Default: None.
        dtype (torch.dtype, optional): The desired data type of the parameters. Default: None.
    Methods:
        forward(input, output_size=None):
            Applies the transposed 2D convolution to the input tensor.
        _output_padding(input, output_size, stride, padding, kernel_size, num_spatial_dims, dilation=None):
            Computes the output padding for the transposed convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        num_rings=None,
        train_index_mat=False,
        inference_accelerate: bool = True,
        force_circular: bool = True,
        dw_fwd: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        if padding_mode != "zeros":
            raise ValueError(
                'Only "zeros" padding mode is supported for {}'.format(
                    self.__class__.__name__
                )
            )
        super(GMR_ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            num_rings,
            train_index_mat,
            inference_accelerate,
            force_circular,
            device,
            dtype,
        )

    # dilation being an optional parameter is for backwards
    # compatibility
    def _output_padding(
        self,
        input,
        output_size,
        stride,
        padding,
        kernel_size,
        num_spatial_dims,
        dilation=None,
    ):
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})".format(
                        num_spatial_dims,
                        input.dim(),
                        num_spatial_dims,
                        num_non_spatial_dims + num_spatial_dims,
                        len(output_size),
                    )
                )

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        (
                            "requested an output size of {}, but valid sizes range "
                            "from {} to {} (for an input of {})"
                        ).format(output_size, min_sizes, max_sizes, input.size()[2:])
                    )

            res = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

    def forward(self, input, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]

        weight_matrix = self._make_weight_matrix(self.weight)

        return F.conv_transpose2d(
            input,
            weight_matrix,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
