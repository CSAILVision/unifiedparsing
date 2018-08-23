
import argparse
import json
import os
import random
from distutils.version import LooseVersion

import cv2
import numpy as np
import torch
from IPython import embed
from scipy.io import loadmat

from broden_dataset.joint_dataset import broden_dataset
from dataset import TrainDataset

colors = loadmat('./data/color150.mat')['colors']


def gen_toy_data():

    records = broden_dataset.training_record_list[0]
    np.random.shuffle(records)
    with open("broden_ade20k_pascal_train_toy.json", "w") as f:
        for i, r in enumerate(records):
            print("i: {}".format(i))
            print("r[nr_part]: {}".format(r['nr_part']))
            if i >= 100:
                break 
            f.write(json.dumps(r) + '\n')
            broden_dataset.resolve_record(r)

    records = broden_dataset.training_record_list[1]
    np.random.shuffle(records)
    with open("broden_os_train_toy.json", "w") as f:
        for i, r in enumerate(records):
            print("i: {}".format(i))
            if i >= 50:
                break 
            f.write(json.dumps(r) + '\n')
            broden_dataset.resolve_record(r)

    embed()


def check_broden_dataset_resolve_records():

    all_record = broden_dataset.record_list['train'][0] + broden_dataset.record_list['train'][1]
    np.random.shuffle(all_record)

    for i, r in enumerate(all_record):
        print(i)
        ret = broden_dataset.resolve_record(r)


def color_encode(label_map):
    label_map = label_map.astype('int')
    labelmap_rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    label_map = label_map % 150
    for label in np.unique(label_map):
        labelmap_rgb += (label_map == label) * np.tile(colors[label], (label_map.shape[0], label_map.shape[1], 1))
    return labelmap_rgb


def expand_transpose_colorencode(label_map):
    label_map = np.array(np.expand_dims(label_map, axis=0), np.uint8)
    label_map = label_map.transpose(1, 2, 0)
    return color_encode(label_map)


def show_batches(batch_dict, use_imshow=False):
    batch_img = batch_dict['img']
    batch_seg_obj = batch_dict['seg_obj']
    batch_valid_obj = batch_dict['valid_objs']
    batch_valid_part = batch_dict["valid_parts"]
    batch_seg_parts = batch_dict["seg_parts"]
    batch_scene_label = batch_dict['scene_label']
    batch_material = batch_dict['seg_material']
    batch_valid_mat = batch_dict['valid_mat']
    nr_batches = batch_img.shape[0]

    for i in range(nr_batches):
        # img
        img = np.array(batch_img[i].transpose(1, 2, 0), np.uint8)
        if use_imshow:
            cv2.imshow('img', img)

        # obj part
        if batch_valid_obj[i]:
            # obj
            seg_obj = np.array(batch_seg_obj[i], np.uint16)
            if use_imshow:
                seg_obj_colored = expand_transpose_colorencode(seg_obj)
                cv2.imshow('obj', seg_obj_colored)
            present_object = np.unique(seg_obj)
            print("objs:")
            for o_l in present_object:
                print("    {}".format(broden_dataset.names["object"][o_l]))
            # part
            if np.sum(batch_valid_part[i]) != 0:
                seg_part = np.array(batch_seg_parts[i], np.uint8)
                if use_imshow:
                    seg_part_colored = expand_transpose_colorencode(seg_part)
                    cv2.imshow('part', seg_part_colored)
                present_part_index = np.unique(seg_part)
                print("all obj-part")
                for j in range(broden_dataset.nr_object_with_part):
                    if not batch_valid_part[i][j]:
                        continue
                    obj_label = broden_dataset.object_with_part[j]
                    print("{}({})".format(obj_label, broden_dataset.names["object"][obj_label]))
                    obj_mask = (seg_obj == obj_label)
                    part_indexs = np.unique(seg_part * obj_mask)
                    for p_i in part_indexs:
                        p_l = broden_dataset.object_part[obj_label][p_i]
                        print("    {}({})".format(p_l, broden_dataset.names["part"][p_l]))
            else:
                print("no part.")
        else:
            print('no obj part.')

        # scene
        scene_label = batch_scene_label[i]
        if scene_label == -1:
            print("no scene label.")
        else:
            print("scene: ")
            print("    {}".format(broden_dataset.names["scene"][scene_label]))

        """
        # texture
        if batch_valid_tex[i]:
            print('texture: ')
            for j, flag in enumerate(batch_texture[i]):
                if flag:
                    print("    {}".format(joint_dataset.names_texture[j]))
        else:
            print('no texture.')
        """

        # material
        if batch_valid_mat[i]:
            seg_material = np.array(batch_material[i], dtype=np.uint8)
            if use_imshow:
                seg_material_colored = expand_transpose_colorencode(seg_material)
                cv2.imshow('material', seg_material_colored)
            print('material:')
            present_material = np.unique(seg_material)
            for m_l in present_material:
                print("    {}".format(broden_dataset.names["material"][m_l]))
        else:
            print('no material.')

        if use_imshow:
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            input('')
            pass


def get_args():
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--weights_encoder', default='',
                        help="weights to finetune net_encoder")
    parser.add_argument('--weights_decoder', default='',
                        help="weights to finetune net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/train.odgt')
    parser.add_argument('--list_val',
                        default='./data/validation.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=8, type=int,
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=5000, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_decoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--fix_bn', default=0, type=int,
                        help='fix bn params')

    # Data related arguments
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=[300, 375, 450, 525, 600], nargs='+', type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.max_iters = args.epoch_iters * args.num_epoch
    args.running_lr_encoder = args.lr_encoder
    args.running_lr_decoder = args.lr_decoder

    args.id += '-' + str(args.arch_encoder)
    args.id += '-' + str(args.arch_decoder)
    args.id += '-ngpus' + str(args.num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgMaxSize' + str(args.imgMaxSize)
    args.id += '-paddingConst' + str(args.padding_constant)
    args.id += '-segmDownsampleRate' + str(args.segm_downsampling_rate)
    args.id += '-LR_encoder' + str(args.lr_encoder)
    args.id += '-LR_decoder' + str(args.lr_decoder)
    args.id += '-epoch' + str(args.num_epoch)
    args.id += '-decay' + str(args.weight_decay)
    args.id += '-fixBN' + str(args.fix_bn)
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    return args


def check_training_dataset():
    args = get_args()
    training_records = broden_dataset.record_list['train']
    for idx_source in [1, 0]:
        dataset_train = TrainDataset(training_records[idx_source], args, batch_per_gpu=args.batch_size_per_gpu)
        for idx_record in range(50):
            print()
            print("**** source idx: {}, record idx: {} **** ".format(idx_source, idx_record))

            print("record: {}".format(training_records[idx_source][idx_record]))
            batch_dict = dataset_train[idx_record]
            show_batches(batch_dict, use_imshow=False)


def check_multi_source_dataset():
    args = get_args()

    from train import create_multi_source_train_data_loader

    multi_source = create_multi_source_train_data_loader(args)

    for i in range(100):
        batch_data, source_idx = next(multi_source)

        print("**** batch idx {} ****".format(i))
        print("source: {}".format(source_idx))

        for idx_gpu in range(args.num_gpus):
            print("gpu idx: {}".format(idx_gpu))
            show_batches(batch_data[idx_gpu])


if __name__ == "__main__":
    # check_broden_dataset_resolve_records()
    # check_training_dataset()
    check_multi_source_dataset()
