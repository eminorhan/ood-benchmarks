"""Evaluate models on shape-texture cue conflict stimuli"""
import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from utils import load_model
from PIL import Image


parser = argparse.ArgumentParser(description='Evaluate models on shape-texture cue-conflict stimuli')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-name', type=str, default='resnext101_32x16d_wsl',
                    choices=['resnext101_32x8d', 'resnext101_32x8d_wsl', 'resnext101_32x16d_wsl',
                             'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl', 'tf_efficientnet_l2_ns',
                             'tf_efficientnet_l2_ns_475', 'tf_efficientnet_b7_ns', 'tf_efficientnet_b6_ns',
                             'tf_efficientnet_b5_ns', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b3_ns',
                             'tf_efficientnet_b2_ns', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b0_ns',
                             'tf_efficientnet_b8', 'tf_efficientnet_b7', 'tf_efficientnet_b6', 'tf_efficientnet_b5',
                             'tf_efficientnet_b4', 'tf_efficientnet_b3', 'tf_efficientnet_b2', 'tf_efficientnet_b1',
                             'tf_efficientnet_b0', 'moco_v2', 'resnet50'],
                    help='evaluated model')
parser.add_argument('--workers', default=4, type=int, help='no of data loading workers')
parser.add_argument('--batch-size', default=36, type=int, help='mini-batch size')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--im-size', default=224, type=int, help='image size')


def validate(model, preprocessing, args):
    from constants import rel_inds

    preds = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        file_list = os.listdir(args.data)
        file_list.sort()

        for file_name in file_list:

            img = Image.open(os.path.join(args.data, file_name))
            proc_img = preprocessing(img)
            proc_img = proc_img.unsqueeze(0)

            if args.gpu is not None:
                proc_img = proc_img.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(proc_img)
            output = output.cpu().numpy().squeeze()

            new_output = -np.inf * np.ones(1000)
            new_output[rel_inds] = output[rel_inds]

            pred = np.argmax(new_output)

            preds.append(pred)

        preds = np.array(preds)

    return preds


def accuracies(labels, preds):
    from constants import conversion_table

    shape_matches = np.zeros(len(preds))
    texture_matches = np.zeros(len(preds))

    for i in range(len(preds)):
        pred = preds[i]
        texture = labels['textures'][i]
        shape = labels['shapes'][i]

        if pred in conversion_table[texture]:
            texture_matches[i] = 1

        if pred in conversion_table[shape]:
            shape_matches[i] = 1

    correct = shape_matches + texture_matches  # texture or shape predictions
    frac_correct = np.mean(correct)
    frac_shape = np.sum(shape_matches) / np.sum(correct)
    frac_texture = np.sum(texture_matches) / np.sum(correct)

    return frac_correct, frac_shape, frac_texture


if __name__ == "__main__":

    args = parser.parse_args()
    model = load_model(args.model_name)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.model_name.startswith('tf_efficientnet'):
        preprocessing = transforms.Compose([
            transforms.Resize(args.im_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    else:
        preprocessing = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    # evaluate on validation set
    preds = validate(model, preprocessing, args)
    labels = np.load('../cueconflict_labels.npz')
    frac_correct, frac_shape, frac_texture = accuracies(labels, preds)

    print('Correct:', frac_correct, 'Shape:', frac_shape, 'Texture:', frac_texture)

    np.savez('shapebias_' + args.model_name + '.npz', frac_correct=frac_correct, frac_shape=frac_shape,
             frac_texture=frac_texture)