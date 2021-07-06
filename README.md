Barlow Twins: Self-Supervised Learning via Redundancy Reduction
---------------------------------------------------------------

<p align="center">
  <img width="500" alt="Screen Shot 2021-04-29 at 6 26 48 AM" src="https://user-images.githubusercontent.com/14848164/120419539-b0fab900-c330-11eb-8536-126ce6ce7b85.png">
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
    <td>73.5%</td>
    <td>91.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/stats.txt">train logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/lincls_0.3/stats.txt">val logs</a></td>
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
python main.py /path/to/imagenet/
```

Training time is approximately 7 days on 16 v100 GPUs.

### Evaluation: Linear Classification

Train a linear probe on the representations learned by Barlow Twins. Freeze the weights of the resnet and use the entire ImageNet training set.

```
python evaluate.py /path/to/imagenet/ /path/to/checkpoint/resnet50.pth --lr-classifier 0.3
```

### Evaluation: Semi-supervised Learning

Train a linear probe on the representations learned by Barlow Twins. Finetune the weights of the resnet and use a subset of the ImageNet training set.

```
python evaluate.py /path/to/imagenet/ /path/to/checkpoint/resnet50.pth --weights finetune --train-perc 1 --epochs 20 --lr-backbone 0.005 --lr-classifier 0.5 --weight-decay 0 --checkpoint-dir ./checkpoint/semisup/
```

### Community Updates

- To use multiple nodes without SLURM, see suggestion by Shoaib: https://github.com/facebookresearch/barlowtwins/issues/13#issuecomment-813126587

- Barlow Twins on CIFAR-10 (PyTorch): https://github.com/IgorSusmelj/barlowtwins

- An intriguing connection of Barlow Twins with the HSIC independence criterion, and some awesome results on small datasets (CIFAR10, STL10 and Tiny-ImageNet) in PyTorch:
https://github.com/yaohungt/Barlow-Twins-HSIC

- A TensorFlow implementation of Barlow Twins on CIFAR-10, by Sayak Paul: https://github.com/sayakpaul/Barlow-Twins-TF

*[Let us know](mailto:jzb@fb.com,ljng@fb.com,imisra@fb.com,yann@fb.com,sdeny@fb.com?subject=[GitHub]%20Barlow%20Twins%20) about all the cool stuff you are able to do with Barlow Twins so that we can advertise it here!*

### License

This project is released under MIT License, which allows commercial use. See [LICENSE](LICENSE) for details.
