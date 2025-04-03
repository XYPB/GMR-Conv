import torch.nn as nn
from .gmr_conv import GMR_Conv2d

def convert_to_gmr_conv(model, kernel_shape='o', train_index_mat=False, gmr_conv_size=None, gmr_groups=None, num_rings=None, force_circular=False):
    '''
    Recursively replace Conv layers to GMR_Conv layers.
    If convolutional layer with stride >= 2 found, add AvgPool2d before it.
    '''
    state = {'counter': 0}

    def _replace_handler(module, state):
        for attr, target in module.named_children():
            if type(target) == nn.Conv2d and target.kernel_size[0] > 1 and target.kernel_size[1] > 1:
                target_padding = target.padding if gmr_conv_size is None else target.dilation[0] * (gmr_conv_size - 1) // 2
                layer = GMR_Conv2d(
                    target.in_channels,
                    target.out_channels,
                    kernel_size=target.kernel_size if gmr_conv_size is None else gmr_conv_size,
                    stride=1,
                    padding=target_padding,
                    dilation=target.dilation,
                    groups=target.groups if gmr_groups is None else gmr_groups,
                    bias=target.bias is not None,
                    num_rings=num_rings,
                    device=target.weight.device,
                    dtype=target.weight.dtype)
                if target.stride[0] > 1 or target.stride[1] > 1:
                    avg_pool = nn.AvgPool2d(target.stride, target.stride)
                    layer = nn.Sequential(avg_pool, layer)
                setattr(module, attr, layer)
                state['counter'] += 1
            _replace_handler(target, state)

    _replace_handler(model, state)