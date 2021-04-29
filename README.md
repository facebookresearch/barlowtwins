Barlow Twins: Self-Supervised Learning via Redundancy Reduction
---------------------------------------------------------------

<p align="center">
  <img width="500" alt="Screen Shot 2021-04-29 at 6 26 48 AM" src="https://user-images.githubusercontent.com/14848164/116538983-3fa19380-a8b6-11eb-8436-4bfe4cdc5d79.png">
</p>

PyTorch implementation of [Barlow Twins](https://arxiv.org/abs/2103.03230).

```
@article{zbontar2021barlow,
  title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
  author={Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, St{\'e}phane},
  journal={arXiv preprint arXiv:2103.03230},
  year={2021}
}
```

### Pretrained Model

<table>
  <tr>
    <th>epochs</th>
    <th>batch size</th>
    <th>acc1</th>
    <th>acc5</th>
    <th colspan="4">download</th>
  </tr>
  <tr>
    <td>1000</td>
    <td>2048</td>
    <td>73.3%</td>
    <td>91.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/barlowtwins/epochs1000_bs2048_lr0.2_lambd0.0051_proj_8192_8192_8192_scale0.024/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/barlowtwins/epochs1000_bs2048_lr0.2_lambd0.0051_proj_8192_8192_8192_scale0.024/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/barlowtwins/epochs1000_bs2048_lr0.2_lambd0.0051_proj_8192_8192_8192_scale0.024/stats.txt">train logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/barlowtwins/epochs1000_bs2048_lr0.2_lambd0.0051_proj_8192_8192_8192_scale0.024/lincls_0.1/stats.txt">val logs</a></td>
  </tr>
</table>

You can choose to download either the weights of the pretrained ResNet-50 network or the full checkpoint, which also contains the weights of the projector network and the state of the optimizer. 

The pretrained model is also available on PyTorch Hub.

```
import torch
model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
```

### Barlow Twins Training

Install PyTorch and download ImageNet by following the instructions in the [requirements](https://github.com/pytorch/examples/tree/master/imagenet#requirements) section of the PyTorch ImageNet training example. The code has been developed for PyTorch version 1.7.1 and torchvision version 0.8.2, but it should work with other versions just as well. 

Our best model is obtained by running the following command:

```
python main.py /path/to/imagenet/ --epochs 1000 --batch-size 2048 --learning-rate 0.2 --lambd 0.0051 --projector 8192-8192-8192 --scale-loss 0.024
```

Training time is approximately 7 days on 16 v100 GPUs.

### Evaluation: Linear Classification

Train a linear probe on the representations learned by Barlow Twins. Freeze the weights of the resnet and use the entire ImageNet training set.

```
python evaluate.py /path/to/imagenet/ /path/to/checkpoint/resnet50.pth --lr-classifier 0.1
```

### Evaluation: Semi-supervised Learning

Train a linear probe on the representations learned by Barlow Twins. Finetune the weights of the resnet and use a subset of the ImageNet training set.

```
python evaluate.py /path/to/imagenet/ /path/to/checkpoint/resnet50.pth --weights finetune --train-perc 1 --epochs 20 --lr-backbone 0.002 --lr-classifier 0.5 --weight-decay 0 --checkpoint-dir ./checkpoint/semisup/
```

### Issues

In order to match the code that was used to develop Barlow Twins, we include an additional parameter, `--scale-loss`, that multiplies the loss by a constant factor. We are working on a version that will not require this parameter.

### Community Updates

- To use multiple nodes without SLURM, see suggestion by Shoaib: https://github.com/facebookresearch/barlowtwins/issues/13#issuecomment-813126587

- Barlow Twins on CIFAR-10: https://github.com/IgorSusmelj/barlowtwins

*[Let us know](mailto:jzb@fb.com,ljng@fb.com,sdeny@fb.com?subject=[GitHub]%20Barlow%20Twins%20) about all the cool stuff you are able to do with Barlow Twins so that we can advertise it here!*

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
