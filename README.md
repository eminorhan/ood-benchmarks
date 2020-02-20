# Out-of-distribution generalization benchmarks for image recognition models
This repository contains code for evaluating the out-of-distribution generalization performance of various image recognition models. Currently, performance on the following benchmarks are evaluated:

* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-C](https://github.com/hendrycks/robustness)
* [ImageNet-P](https://github.com/hendrycks/robustness)
* [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)
* [Stylized ImageNet](https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512)
* Adversarial robustness

The following models are currently evaluated:

* `resnext101_32x48d_wsl`: the largest ResNeXt-WSL model released by Facebook AI (res: 224)
* `resnext101_32x32d_wsl`: second largest ResNeXt-WSL model released by Facebook AI (res: 224)
* `resnext101_32x16d_wsl`: third largest ResNeXt-WSL model released by Facebook AI (res: 224)
* `resnext101_32x8d_wsl`: fourth largest ResNeXt-WSL model released by Facebook AI (res: 224)
* `resnext101_32x8d`: ImageNet pre-trained ResNeXt model (res: 224)
* `tf_efficientnet_l2_ns`: the largest noisy student model released by Google AI (res: 800)
* `tf_efficientnet_l2_ns_475`: lower resolution version of the above (res: 475)
* `tf_efficientnet_b7_ns`: noisy student model with B7 backbone (res: 600)
* `tf_efficientnet_b6_ns`: noisy student model with B6 backbone (res: 528)
* `tf_efficientnet_b5_ns`: noisy student model with B5 backbone (res: 456)
* `tf_efficientnet_b4_ns`: noisy student model with B4 backbone (res: 380)
* `tf_efficientnet_b3_ns`: noisy student model with B3 backbone (res: 300)
* `tf_efficientnet_b2_ns`: noisy student model with B2 backbone (res: 260)
* `tf_efficientnet_b1_ns`: noisy student model with B1 backbone (res: 240)
* `tf_efficientnet_b0_ns`: noisy student model with B0 backbone (res: 224)
* `tf_efficientnet_b8`: RandAugment trained model with B8 backbone (res: 672)
* `tf_efficientnet_b7`: RandAugment trained model with B7 backbone (res: 600)
* `tf_efficientnet_b6`: RandAugment trained model with B6 backbone (res: 528)
* `tf_efficientnet_b5`: RandAugment trained model with B5 backbone (res: 456)
* `tf_efficientnet_b4`: RandAugment trained model with B4 backbone (res: 380)
* `tf_efficientnet_b3`: RandAugment trained model with B3 backbone (res: 300)
* `tf_efficientnet_b2`: RandAugment trained model with B2 backbone (res: 260)
* `tf_efficientnet_b1`: RandAugment trained model with B1 backbone (res: 240)
* `tf_efficientnet_b0`: RandAugment trained model with B0 backbone (res: 224)

All simulation results reported on this page can be found in the [`results`](https://github.com/eminorhan/ood-benchmarks/tree/master/results) folder. 

## Requirements
The code was written and tested with:

* torch == 1.3.0
* torchvision == 0.4.0
* foolbox == 1.8.0

Other versions may or may not work. In addition, you will need to download the datasets listed above in order to replicate the results. Please let me know if you encounter any issues.

## Results
| Model | IN | IN-A | IN-C | IN-P | Stylized IN | IN-Sketch | Adv. acc. |
| ----- |:--:|:----:|:----:|:----:|:-----------:|:---------:|:---------:|
| `resnext101_32x48d_wsl`     | TBD | TBD | TBD | TBD | 42.8 | **59.1** | 29.7 |
| `resnext101_32x32d_wsl`     | TBD | TBD | TBD | TBD | 40.6 | 58.6 | 30.6 |
| `resnext101_32x16d_wsl`     | TBD | TBD | TBD | TBD | 42.7 | 57.9 | **40.7** |
| `resnext101_32x8d_wsl`      | TBD | TBD | TBD | TBD | 39.1 | 55.2 | 34.4 |
| `resnext101_32x8d`          | TBD | TBD | TBD | TBD | 25.9 | 28.6 | 0.0 |
| `tf_efficientnet_l2_ns`     | TBD | TBD | TBD | TBD | 39.0 | 52.7 | TBD |
| `tf_efficientnet_l2_ns_475` | TBD | TBD | TBD | TBD | **61.8** | 53.6 | TBD |
| `tf_efficientnet_b7_ns`     | TBD | TBD | TBD | TBD | 44.1 | 48.3 | 2.9 |
| `tf_efficientnet_b6_ns`     | TBD | TBD | TBD | TBD | 35.1 | 48.1 | 5.1 |
| `tf_efficientnet_b5_ns`     | TBD | TBD | TBD | TBD | 32.3 | 45.1 | 3.7 |
| `tf_efficientnet_b4_ns`     | TBD | TBD | TBD | TBD | 29.5 | 43.2 | 9.0 |
| `tf_efficientnet_b3_ns`     | TBD | TBD | TBD | TBD | 26.2 | 39.4 | 6.8 |
| `tf_efficientnet_b2_ns`     | TBD | TBD | TBD | TBD | 25.4 | 36.1 | 5.6 |
| `tf_efficientnet_b1_ns`     | TBD | TBD | TBD | TBD | 27.7 | 34.0 | 5.4 |
| `tf_efficientnet_b0_ns`     | TBD | TBD | TBD | TBD | 24.3 | 28.9 | 2.7 |
| `tf_efficientnet_b8`     | TBD | TBD | TBD | TBD | 31.3 | 40.3 | TBD |
| `tf_efficientnet_b7`     | TBD | TBD | TBD | TBD | 31.7 | 38.7 | TBD |
| `tf_efficientnet_b6`     | TBD | TBD | TBD | TBD | 18.6 | 32.4 | TBD |
| `tf_efficientnet_b5`     | TBD | TBD | TBD | TBD | 26.9 | 36.4 | TBD |
| `tf_efficientnet_b4`     | TBD | TBD | TBD | TBD | 22.8 | 32.7 | TBD |
| `tf_efficientnet_b3`     | TBD | TBD | TBD | TBD | 22.5 | 31.8 | TBD |
| `tf_efficientnet_b2`     | TBD | TBD | TBD | TBD | 29.0 | 29.3 | TBD |
| `tf_efficientnet_b1`     | TBD | TBD | TBD | TBD | 25.2 | 28.2 | TBD |
| `tf_efficientnet_b0`     | TBD | TBD | TBD | TBD | 26.3 | 26.5 | TBD |

In my experience, it is possible to get slightly different numbers from those reported above (up to ~1\%) using different pre-processing strategies, but the overall patterns should be robust to these changes. Adversarial accuracy refers to top-1 accuracy against white-box PGD attacks with a normalized perturbation size of 0.06 in the *l*<sub>inf</sub> metric (see [my paper](https://arxiv.org/abs/1907.07640) for more details). Unfortunately, I didn't have enough GPU RAM to run atttacks against the largest noisy student models, `tf_efficientnet_l2_ns` and `tf_efficientnet_l2_ns_475`, but I don't expect the results to be much different from the smaller noisy student models.

## Replication
For replication, please see the shell scripts in [`scripts`](https://github.com/eminorhan/ood-benchmarks/tree/master/scripts) that were used to obtain the results reported on this page. 

## Acknowledgments
The code here utilizes code and stimuli from the [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) repository by Robert Geirhos, the [robustness](https://github.com/hendrycks/robustness) and [natural adversarial examples](https://github.com/hendrycks/natural-adv-examples) repositories by Dan Hendrycks, and the [ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet) from PyTorch. I am grateful to the authors of [Mahajan et al. (2018)](https://arxiv.org/abs/1805.00932) and [Xie et al. (2019)](https://arxiv.org/abs/1911.04252) for making their pre-trained models publicly available. I am also grateful to Ross Wightman for porting Google's EfficientNet models to PyTorch (see his repo [here](https://github.com/rwightman/gen-efficientnet-pytorch)), which were used in the experiments reported here.
