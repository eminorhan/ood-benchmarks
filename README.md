## Out-of-distribution generalization benchmarks for image recognition models
This repository contains code for evaluating the out-of-distribution generalization performance of various image recognition models. Currently, performance on the following benchmarks are evaluated:

* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-C](https://github.com/hendrycks/robustness)
* [ImageNet-P](https://github.com/hendrycks/robustness)
* [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)
* [Stylized ImageNet](https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512)
* Adversarial robustness

The following models are currently evaluated:

* `moco_v2`: State-of-the-art self-supervised [MoCo v2](https://github.com/facebookresearch/moco) model with ResNet-50 backbone (res: 224)
* `resnet50`: Supervised ResNet-50 model (res: 224)
* `resnext101_32x32d_wsl`: second largest ResNeXt-WSL model released by Facebook AI (res: 224)
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

## Results
| Model | IN | IN-A | IN-C | IN-P | Stylized IN | IN-Sketch | Adv. acc. |
| ----- |:--:|:----:|:----:|:----:|:-----------:|:---------:|:---------:|
| `moco_v2`     | 70.2 | 3.6 | TBD | TBD | 29.4 | 18.8 | 0.1 |
| `resnet50`    | 76.1 | 0.0 | TBD | TBD | 22.1 | 23.0 | 0.0 |
| `resnext101_32x48d_wsl`     | TBD | 61.0 | TBD | TBD | 42.8 | **59.1** | 29.7 |
| `resnext101_32x32d_wsl`     | TBD | 58.1 | TBD | TBD | 40.6 | 58.6 | 30.6 |
| `resnext101_32x16d_wsl`     | TBD | 53.1 | TBD | TBD | 42.7 | 57.9 | **40.7** |
| `resnext101_32x8d_wsl`      | TBD | 45.4 | TBD | TBD | 39.1 | 55.2 | 34.4 |
| `resnext101_32x8d`          | TBD | 10.2 | TBD | TBD | 25.9 | 28.6 | 0.0 |
| `tf_efficientnet_l2_ns`     | TBD | **84.9** | TBD | TBD | 39.0 | 52.7 | TBD |
| `tf_efficientnet_l2_ns_475` | TBD | 83.4 | TBD | TBD | **61.8** | 53.6 | TBD |
| `tf_efficientnet_b7_ns`     | TBD | 66.5 | TBD | TBD | 44.1 | 48.3 | 2.9 |
| `tf_efficientnet_b6_ns`     | TBD | 61.5 | TBD | TBD | 35.1 | 48.1 | 5.1 |
| `tf_efficientnet_b5_ns`     | TBD | 58.9 | TBD | TBD | 32.3 | 45.1 | 3.7 |
| `tf_efficientnet_b4_ns`     | TBD | 48.9 | TBD | TBD | 29.5 | 43.2 | 9.0 |
| `tf_efficientnet_b3_ns`     | TBD | 32.7 | TBD | TBD | 26.2 | 39.4 | 6.8 |
| `tf_efficientnet_b2_ns`     | TBD | 20.9 | TBD | TBD | 25.4 | 36.1 | 5.6 |
| `tf_efficientnet_b1_ns`     | TBD | 17.0 | TBD | TBD | 27.7 | 34.0 | 5.4 |
| `tf_efficientnet_b0_ns`     | TBD | 10.3 | TBD | TBD | 24.3 | 28.9 | 2.7 |
| `tf_efficientnet_b8`     | TBD | 48.0 | TBD | TBD | 31.3 | 40.3 | 0.5 |
| `tf_efficientnet_b7`     | TBD | 42.5 | TBD | TBD | 31.7 | 38.7 | 0.6 |
| `tf_efficientnet_b6`     | TBD | 34.2 | TBD | TBD | 18.6 | 32.4 | 0.3 |
| `tf_efficientnet_b5`     | TBD | 30.5 | TBD | TBD | 26.9 | 36.4 | 0.9 |
| `tf_efficientnet_b4`     | TBD | 24.7 | TBD | TBD | 22.8 | 32.7 | 0.5 |
| `tf_efficientnet_b3`     | TBD | 15.1 | TBD | TBD | 22.5 | 31.8 | 0.6 |
| `tf_efficientnet_b2`     | TBD | 8.7 | TBD | TBD | 29.0 | 29.3 | 0.5 |
| `tf_efficientnet_b1`     | TBD | 7.2 | TBD | TBD | 25.2 | 28.2 | 0.5 |
| `tf_efficientnet_b0`     | TBD | 4.8 | TBD | TBD | 26.3 | 26.5 | 0.3 |

**Notes:** 

1. In my experience, it is possible to get slightly different numbers from those reported above (up to ~1\%) using different pre-processing strategies, but the overall patterns should be robust to these changes. 

2. Adversarial accuracy refers to top-1 accuracy against white-box PGD attacks with a normalized perturbation size of 0.06 in the *l*<sub>inf</sub> metric (see [my paper](https://arxiv.org/abs/1907.07640) for more details). 

3. Unfortunately, I didn't have enough GPU RAM to run white-box atttacks against the largest noisy student models, `tf_efficientnet_l2_ns` and `tf_efficientnet_l2_ns_475`, using their native image resolution, but I don't expect the results to be much different from the smaller noisy student models. 

4. Note that the noisy student and WSL models are not directly comparable with respect to adversarial accuracy, since the noisy student models use larger images and it is significantly easier to run successful adversarial attacks with larger images.

5. However, I was able to run adversarial attacks against the `tf_efficientnet_l2_ns_475` model using images of size 224x224. Although this is not ideal (because the model was trained with images of a different size), this allows us to make a rough comparison between the WSL models and the noisy student models with respect to adversarial robustness. The `tf_efficientnet_l2_ns_475` model had an adversarial accuracy of 22.3\% in this setting, demonstrating that the observed improvement in adversarial accuracy that comes with training with extra data is very much real and is not specific to the WSL models.

6. I didn't have enough time for training the linear ImageNet classifier on top of MoCo v2 for 100 epochs (as recommended in the original [MoCo repository](https://github.com/facebookresearch/moco)), so I trained it for 15 epochs only, reducing the learning rate after the 12th and 14th epochs. The top-1 accuracy of the MoCo v2 model used here thus is slightly lower than the accuracy reported in the MoCo repository (70.2 vs. 71.1), but I don't expect that this discrepancy fundamentally affects the results above. 

## Discussion
1. More training data improves robustness across the board (including adversarial robustness).

2. Large capacity models are crucial for bringing out the benefits of extra training data.

3. On the ImageNet-Sketch benchmark, WSL models perform slightly better than the noisy student models. I suspect that this may be because the Instagram data that WSL models were trained with might already contain such sketch-like images. This raises an important concern with both of these models. The large scale datasets these models were trained with (Instagram-1B and JFT-300M) are both private, so we don't exactly know what kind of images they contain. This means that we also don't exactly know to what extent these "out-of-distribution" benchmarks are really out of distribution for these models. 

4. On the stylized ImageNet benchmark, the `tf_efficientnet_l2_ns_475` model achieves a respectable shape-based prediction rate of 61.8\%. Although this is still far from the shape-based prediction rate for humans (>90\%), it suggests to me that we might be able to overcome the strong texture bias of image recognition models by pushing the noisy student approach to even larger datasets.

5. State of the art self-supervised models don't have fundamentally different generalization properties than supervised models using the same architecture (compare `moco_v2` and `resnet50`). This could be both good news and bad news: on the one hand, it's good that the new generation self-supervised learning methods yield models that perform similarly to supervised models across a broad range of generalization benchmarks, but on the other hand, these methods don't give us a magic bullet that will solve the generalization problems of standard image recognition models. It's also important to note that these new generation self-supervised learning methods are computationally much more expensive than supervised learning, generally requiring more extensive data augmentation, longer training, and larger batch sizes than supervised learning methods.

## Replication
For replication, please see the shell scripts in [`scripts`](https://github.com/eminorhan/ood-benchmarks/tree/master/scripts) that were used to obtain the results reported on this page. 

## Requirements
The code was written and tested with:

* torch == 1.3.0
* torchvision == 0.4.0
* foolbox == 1.8.0

Other versions may or may not work. In addition, you will need to download the datasets listed above in order to replicate the results. Please let me know if you encounter any issues.

## Acknowledgments
The code here utilizes stimuli from Haohan Wang's [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch) repository, code and stimuli from the [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) repository by Robert Geirhos, the [robustness](https://github.com/hendrycks/robustness) and [natural adversarial examples](https://github.com/hendrycks/natural-adv-examples) repositories by Dan Hendrycks, and the [ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet) from PyTorch. I am grateful to the authors of [Mahajan et al. (2018)](https://arxiv.org/abs/1805.00932), [Xie et al. (2019)](https://arxiv.org/abs/1911.04252), [He et al. (2019)](https://arxiv.org/abs/1911.05722), and [Chen et al. (2020)](https://arxiv.org/abs/2003.04297) for making their pre-trained models publicly available. I am also grateful to Ross Wightman for porting Google's EfficientNet models to PyTorch (see his repo [here](https://github.com/rwightman/gen-efficientnet-pytorch)), which were used in the experiments reported here.
