"""Evaluate models on ImageNet-A"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from calibration_tools import show_calibration_results
from utils import load_model, ShortEdgeCenterCrop
from PIL import Image


parser = argparse.ArgumentParser(description='Evaluate models on ImageNet-A')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-name', type=str, default='resnext101_32x16d_wsl',
                    choices=['resnext101_32x8d', 'resnext101_32x8d_wsl', 'resnext101_32x16d_wsl',
                             'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl', 'tf_efficientnet_l2_ns',
                             'tf_efficientnet_l2_ns_475', 'tf_efficientnet_b7_ns', 'tf_efficientnet_b6_ns',
                             'tf_efficientnet_b5_ns', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b3_ns',
                             'tf_efficientnet_b2_ns', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b0_ns',
                             'tf_efficientnet_b8', 'tf_efficientnet_b7', 'tf_efficientnet_b6', 'tf_efficientnet_b5',
                             'tf_efficientnet_b4', 'tf_efficientnet_b3', 'tf_efficientnet_b2', 'tf_efficientnet_b1',
                             'tf_efficientnet_b0'],
                    help='evaluated model')
parser.add_argument('--workers', default=4, type=int, help='no of data loading workers')
parser.add_argument('--batch-size', default=36, type=int, help='mini-batch size')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--im-size', default=224, type=int, help='image size')


def to_np(x):
    return x.data.to('cpu').numpy()


def get_net_results(net, loader):
    from constants import indices_in_1k

    confidence = []
    correct = []

    net.eval()

    num_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()

            output = net(data)[:, indices_in_1k]

            # accuracy
            pred = output.data.max(1)[1]
            num_correct += pred.eq(target.data).sum().item()

            confidence.extend(to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist())
            pred = output.data.max(1)[1]
            correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())

    return num_correct / len(loader.dataset), confidence.copy(), correct.copy()


if __name__ == "__main__":

    args = parser.parse_args()
    model = load_model(args.model_name)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.model_name.startswith('resnext'):
        combined_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
        ])
    elif args.model_name.startswith('tf_efficientnet'):
        combined_transform = transforms.Compose([
            ShortEdgeCenterCrop(),
            transforms.Resize(args.im_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise ValueError('Model not available.')

    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data, combined_transform),
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # evaluate
    acc, test_confidence, test_correct = get_net_results(model, val_loader)

    print('ImageNet-A Accuracy (%):', round(100 * acc, 4))

    cal_err, aur = show_calibration_results(np.array(test_confidence), np.array(test_correct))

    np.savez('imaageneta_' + args.model_name + '.npz', acc=acc, cal_err=cal_err, aur=aur)