"""Evaluate models on ImageNet-C"""
import argparse
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import load_model, validate


parser = argparse.ArgumentParser(description='Evaluate models on ImageNet-C')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-name', type=str, default='resnext101_32x16d_wsl',
                    choices=['resnext101_32x8d', 'resnext101_32x8d_wsl', 'resnext101_32x16d_wsl',
                             'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl'], help='evaluated model')
parser.add_argument('--workers', default=4, type=int, help='no of data loading workers')
parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')


if __name__ == "__main__":

    args = parser.parse_args()

    model = load_model(args.model_name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                   'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                   'jpeg_compression']

    distortion_errors = []

    for distortion_name in distortions:

        errs = []

        for severity in range(1, 6):
            print(distortion_name, str(severity))

            valdir = args.data + distortion_name + '/' + str(severity)
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            # evaluate on validation set
            acc1 = validate(val_loader, model, args)
            errs.append(100. - acc1)

        print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 1. * np.mean(errs)))
        distortion_errors.append(np.mean(errs))

    np.save('imagenetc_errors_' + args.model_name + '.npy', np.array(distortion_errors))