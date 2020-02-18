"""Evaluate model on ImageNet validation data"""
import os
import argparse
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import load_model, validate
from PIL import Image


parser = argparse.ArgumentParser(description='Evaluate model on ImageNet validation data')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-name', type=str, default='resnext101_32x16d_wsl',
                    choices=['resnext101_32x8d', 'resnext101_32x8d_wsl', 'resnext101_32x16d_wsl',
                             'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl', 'tf_efficientnet_l2_ns',
                             'tf_efficientnet_l2_ns_475'], help='evaluated model')
parser.add_argument('--workers', default=4, type=int, help='no of data loading workers')
parser.add_argument('--batch-size', default=36, type=int, help='mini-batch size')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--im-size', default=224, type=int, help='image size')


if __name__ == "__main__":

    args = parser.parse_args()
    model = load_model(args.model_name)
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.model_name.startswith('resnext'):
        combined_transform = transforms.Compose([
                transforms.Resize(256, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
        ])
    elif args.model_name.startswith('tf_efficientnet'):
        combined_transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Resize(args.im_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise ValueError('Model not available.')

    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, combined_transform),
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # evaluate on validation set
    acc1 = validate(val_loader, model, args)

    np.save('sketch_' + args.model_name + '.npy', acc1)