# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision.models.resnet import resnet50 as _resnet50

dependencies = ['torch', 'torchvision']


def resnet50(pretrained=True, **kwargs):
    model = _resnet50(pretrained=False, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model
