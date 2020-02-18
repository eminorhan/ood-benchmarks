# Out-of-distribution generalization benchmarks for image recognition models
This repository contains code for evaluating the out-of-distribution generalization performance of various image recognition models. Currently, performance on the following benchmarks are evaluated:

* [ImageNet-C](https://github.com/hendrycks/robustness)
* [ImageNet-P](https://github.com/hendrycks/robustness)
* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)
* [Stylized ImageNet](https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512)
* Adversarial robustness

The following models are currently evaluated:

* `resnext101_32x48d_wsl`: the largest ResNeXt-WSL model released by Facebook AI Research.
* `tf_efficientnet_l2_ns`: the largest noisy student model released by Google AI (res: 800).
* `tf_efficientnet_l2_ns_475`: lower resolution version of the above (res: 475).
* `tf_efficientnet_b7_ns`: noisy student model with B7 backbone (res: 600).
* `tf_efficientnet_b6_ns`: noisy student model with B6 backbone (res: 528).
* `tf_efficientnet_b5_ns`: noisy student model with B5 backbone (res: 456).
* `tf_efficientnet_b4_ns`: noisy student model with B4 backbone (res: 380).
* `tf_efficientnet_b3_ns`: noisy student model with B3 backbone (res: 300).
* `tf_efficientnet_b2_ns`: noisy student model with B2 backbone (res: 260).
* `tf_efficientnet_b1_ns`: noisy student model with B1 backbone (res: 240).
* `tf_efficientnet_b0_ns`: noisy student model with B0 backbone (res: 224).

All simulation results reported on this page can be found in the [`results`](https://github.com/eminorhan/ood-benchmarks/tree/master/results) folder. 

## Requirements
The code was written and tested with:

* torch == 1.3.0
* torchvision == 0.4.0
* foolbox == 1.8.0

Other versions may or may not work. In addition, you will need to download the datasets listed above in order to replicate the results.

## Results
| Model | IN | IN-A | IN-C | IN-P | Stylized IN | IN-Sketch | Adv. acc. |
| ----- |:--:|:----:|:----:|:----:|:-----------:|:---------:|:---------:|
| `resnext101_32x48d_wsl`     | TBD | TBD | TBD | TBD | 42.8 | 59.1 | TBD |
| `tf_efficientnet_l2_ns`     | TBD | TBD | TBD | TBD | 39.0 | 52.7 | TBD |
| `tf_efficientnet_l2_ns_475` | TBD | TBD | TBD | TBD | 61.8 | 53.6 | TBD |
| `tf_efficientnet_b7_ns`     | TBD | TBD | TBD | TBD | 44.1 | 48.3 | TBD |
| `tf_efficientnet_b6_ns`     | TBD | TBD | TBD | TBD | 35.1 | TBD  | TBD |
| `tf_efficientnet_b5_ns`     | TBD | TBD | TBD | TBD | 32.3 | TBD  | TBD |
| `tf_efficientnet_b4_ns`     | TBD | TBD | TBD | TBD | 29.5 | TBD  | TBD |
| `tf_efficientnet_b3_ns`     | TBD | TBD | TBD | TBD | 26.2 | TBD  | TBD |
| `tf_efficientnet_b2_ns`     | TBD | TBD | TBD | TBD | 25.4 | TBD  | TBD |
| `tf_efficientnet_b1_ns`     | TBD | TBD | TBD | TBD | 27.7 | TBD  | TBD |
| `tf_efficientnet_b0_ns`     | TBD | TBD | TBD | TBD | 24.3 | TBD  | TBD |

## Replication
For replication, please see the shell scripts in [`scripts`](https://github.com/eminorhan/ood-benchmarks/tree/master/scripts) that were used to obtain the results reported on this page. 

## Acknowledgments
The code here utilizes code and stimuli from the [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) repository by Robert Geirhos, the [robustness](https://github.com/hendrycks/robustness) and [natural adversarial examples](https://github.com/hendrycks/natural-adv-examples) repositories by Dan Hendrycks, and the [ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet) from PyTorch. We are grateful to the authors of [Mahajan et al. (2018)](https://arxiv.org/abs/1805.00932) and [Xie et al. (2019)](https://arxiv.org/abs/1911.04252) for making their pre-trained models publicly available. We are also grateful to Ross Wightman for porting Google's EfficientNet models to PyTorch (see his repo [here](https://github.com/rwightman/gen-efficientnet-pytorch)), which were used in the experiments reported here.
