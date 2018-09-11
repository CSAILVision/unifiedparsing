# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy, mark_volatile
import lib.utils.data as torchdata
import cv2
from broden_dataset_utils.joint_dataset import broden_dataset
from utils import maskrcnn_colorencode, remove_small_mat


def visualize_result(data, preds, args):

    np.random.seed(233)
    color_list = np.random.rand(1000, 3) * .7 + .3

    # image
    img = data['img_ori']
    cv2.imwrite(os.path.join(args.result, "original_image.jpg"), img)

    # object
    object_result = preds['object']
    object_result_colored = maskrcnn_colorencode(img, object_result, color_list)
    cv2.imwrite(os.path.join(args.result, "object_result.png"), object_result_colored)

    # part
    img_part_pred, part_result = img.copy(), preds['part']
    valid_object = np.zeros_like(object_result)
    present_obj_labels = np.unique(object_result)
    for obj_part_index, object_label in enumerate(broden_dataset.object_with_part):
        if object_label not in present_obj_labels:
            continue
        object_mask = (object_result == object_label)
        valid_object += object_mask
        part_result_masked = part_result[obj_part_index] * object_mask
        present_part_label = np.unique(part_result_masked)
        if len(present_part_label) == 1:
            continue
        img_part_pred = maskrcnn_colorencode(
            img_part_pred, part_result_masked + object_mask, color_list)
    cv2.imwrite(os.path.join(args.result, "part_result.png"), img_part_pred)

    # scene
    print("scene shape: {}".format(preds['scene'].shape))
    scene_top5 = np.argsort(-preds['scene'])[:5]
    with open(os.path.join(args.result, "scene.txt"), 'w') as f:
        f.write("scene pred:\n")
        scene_info = ["{}({}) {:.4f}".format(
            l, broden_dataset.names['scene'][l], preds['scene'][l])
            for l in scene_top5]
        f.write("\n".join(scene_info))

    # material
    material_result = preds['material']
    img_material_result = maskrcnn_colorencode(
        img, remove_small_mat(material_result * (valid_object > 0), object_result), color_list)
    cv2.imwrite("material_result.png", img_material_result)


def test(segmentation_module, loader, args):
    segmentation_module.eval()

    for i, data in enumerate(loader):
        # process data
        data = data[0]
        seg_size = data['img_ori'].shape[0:2]

        with torch.no_grad():
            pred_ms = {}
            for k in ['object', 'material']:
                pred_ms[k] = torch.zeros(1, args.nr_classes[k], *seg_size)
            pred_ms['part'] = []
            for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                n_part = len(broden_dataset.object_part[object_label])
                pred_ms['part'].append(torch.zeros(1, n_part, *seg_size))
            pred_ms['scene'] = torch.zeros(1, args.nr_classes['scene'])

            for img in data['img_data']:
                # forward pass
                feed_dict = async_copy_to({"img": img}, args.gpu_id)
                pred = segmentation_module(feed_dict, seg_size=seg_size)
                for k in ['scene', 'object', 'material']:
                    pred_ms[k] = pred_ms[k] + pred[k].cpu() / len(args.imgSize)
                for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                    pred_ms['part'][idx_part] += pred['part'][idx_part].cpu() / len(args.imgSize)

            pred_ms['scene'] = pred_ms['scene'].squeeze(0)
            for k in ['object', 'material']:
                _, p_max = torch.max(pred_ms[k].cpu(), dim=1)
                pred_ms[k] = p_max.squeeze(0)
            for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                _, p_max = torch.max(pred_ms['part'][idx_part].cpu(), dim=1)
                pred_ms['part'][idx_part] = p_max.squeeze(0)
            pred_ms = as_numpy(pred_ms)

        visualize_result(data, pred_ms, args)

        print('[{}] iter {}'
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i))


def main(args):
    torch.cuda.set_device(args.gpu_id)

    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        nr_classes=args.nr_classes,
        weights=args.weights_decoder,
        use_softmax=True)
    
    segmentation_module = SegmentationModule(net_encoder, net_decoder)
    segmentation_module.cuda()

    # Dataset and Loader
    list_test = [{'fpath_img': args.test_img}]
    dataset_val = TestDataset(
        list_test, args, max_sample=args.num_val)
    loader_val = torchdata.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    # Main loop
    test(segmentation_module, loader_val, args)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--test_img', required=True)
    parser.add_argument('--model_path', required=True,
                        help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    # Misc arguments
    parser.add_argument('--result', default='.',
                        help='folder to output visualization results')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='gpu_id for evaluation')

    args = parser.parse_args()
    print(args)

    nr_classes = broden_dataset.nr.copy()
    nr_classes['part'] = sum(
        [len(parts) for obj, parts in broden_dataset.object_part.items()])
    args.nr_classes = nr_classes

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.model_path,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.model_path,
                                        'decoder' + args.suffix)

    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
