import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

from datasets import LeafDataset, LeafDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--dataset_mode', type=str, default='segmented')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')

    parser.add_argument('--display_freq', type=int, default=1)

    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth')

    # Common options
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of classes in semantic map')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # Options specific to GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # Options specific to ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')

    opt = parser.parse_args()
    return opt


def test(opt, seg, gmm, alias):
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss.cuda()

    test_dataset = LeafDataset(opt)
    test_loader = LeafDataLoader(opt, test_dataset)

    success_count = 0

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['healthy_img_name']  # Assuming 'healthy_img_name' corresponds to image names
            c_names = inputs['diseased_img_name']  # Assuming 'diseased_img_name' corresponds to image names

            img_agnostic = inputs['healthy_img'].cuda()
            parse_agnostic = inputs['diseased_img'].cuda()

            # Process inputs through your models
            # Example:
            output = seg(img_agnostic)
            output = gmm(output, parse_agnostic)
            output = alias(output, img_agnostic, parse_agnostic)

            # Example of saving images
            save_path = os.path.join(opt.save_dir, opt.name)
            save_images(output, img_names, save_path)  # Adjust parameters as per your dataset

            # Check if images were successfully generated
            if os.path.exists(save_path):
                success_count += 1
                print(f"Generated images successfully for step {i + 1}.")
            else:
                print(f"Failed to generate images for step {i + 1}.")

            if (i + 1) % opt.display_freq == 0:
                print("Processed step: {}".format(i + 1))

    print(f"Successfully generated {success_count} images out of {len(test_loader)} steps.")

def main():
    opt = get_opt()
    print(opt)

    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))

    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

    seg.cuda().eval()
    gmm.cuda().eval()
    alias.cuda().eval()
    test(opt, seg, gmm, alias)


if __name__ == '__main__':
    main()
