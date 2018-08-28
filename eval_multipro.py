# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import ValDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, intersection_union_part
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy, mark_volatile
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm

from broden_dataset.joint_dataset import broden_dataset


def visualize_result(data, preds, args):
    colors = loadmat('data/color150.mat')['colors']
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(preds, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    cv2.imwrite(os.path.join(args.result,
                img_name.replace('.jpg', '.png')), im_vis)


def get_metrics(pred, data):
    metric = {}

    # scene
    metric['scene'] = {}
    metric['scene']['gt'] = data['scene_label']
    if metric['scene']['gt'] != -1:
        scene_t1 = np.argmax(pred['scene'])
        metric['scene']['top1'] = (scene_t1 == metric['scene']['gt'])

    # object, part
    metric['valid_object'] = data['valid_object']
    if metric['valid_object']:

        # object
        metric['object'] = {}
        object_gt, object_pred = data['seg_object'], pred['object']

        metric["object"]["acc"] = ((object_gt == object_pred) * object_gt > 0).sum()  # ignore 0
        metric["object"]["pixel"] = (object_gt > 0).sum()  # ignore 0

        metric["object"]["inter"], metric["object"]["uni"] = intersectionAndUnion(
            object_pred, object_gt, broden_dataset.nr['object'] - 1)  # ignore 0

        # parts
        metric['part'] = []
        metric["valid_part"] = data["valid_part"]
        for object_part_idx, object_label in enumerate(broden_dataset.object_with_part):

            if not data['valid_part'][object_part_idx]:
                metric["part"].append(None)
                continue

            # NOTE: nr part include background 0.
            nr_part = len(broden_dataset.object_part[object_label])

            object_pred_mask = (object_pred == object_label)
            object_gt_mask = (object_gt == object_label)
            parts_gt = data["seg_part"] * object_gt_mask
            parts_pred = pred['part'][object_part_idx] * object_pred_mask

            _result = {}
            _result["acc"] = ((parts_gt == parts_pred) * (parts_gt > 0)).sum()  # ignore 0
            _result["pixel"] = (parts_gt > 0).sum()  # ignore 0
            _result["inter"], _result["uni"] = intersection_union_part(
                parts_pred, parts_gt, nr_part)  # mIoU-bg, include background 0

            metric["part"].append(_result)

    # material
    metric['valid_material'] = data['valid_material']
    if metric['valid_material']:
        metric['material'] = {}
        material_gt, material_pred = data['seg_material'], pred['material']

        metric["material"]["acc"] = ((material_gt == material_pred) * material_gt > 0).sum()  # ignore 0
        metric["material"]["pixel"] = (material_gt > 0).sum()  # ignore 0

        metric["material"]["inter"], metric["material"]["uni"] = intersectionAndUnion(
            material_pred, material_gt, broden_dataset.nr['material'] - 1)  # ignore 0

    return metric


def evaluate(segmentation_module, loader, args, dev_id, result_queue):
    segmentation_module.eval()

    for i, data_torch in enumerate(loader):
        data_torch = data_torch[0]  # TODO(LYC):: support batch size > 1
        data_np = as_numpy(data_torch)
        seg_size = data_np['seg_object'].shape[0:2]

        with torch.no_grad():
            pred_ms = {}
            for k in ['object', 'material']:
                pred_ms[k] = torch.zeros(1, args.nr_classes[k], *seg_size)
            pred_ms['part'] = []
            for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                n_part = len(broden_dataset.object_part[object_label])
                pred_ms['part'].append(torch.zeros(1, n_part, *seg_size))
            pred_ms['scene'] = torch.zeros(1, args.nr_classes['scene'])

            for img in data_torch['img_resized_list']:
                # forward pass
                feed_dict = async_copy_to({"img": img.unsqueeze(0)}, dev_id)
                pred = segmentation_module(feed_dict, seg_size=seg_size)
                for k in ['scene', 'object', 'material']:
                    pred_ms[k] = pred_ms[k] + pred[k].cpu() / len(args.imgSize)
                for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                    pred_ms['part'][idx_part] += pred['part'][idx_part].cpu() / len(args.imgSize)

            for k in ['scene', 'object', 'material']:
                _, p_max = torch.max(pred_ms[k].cpu(), dim=1)
                pred_ms[k] = p_max.squeeze(0)
            for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                _, p_max = torch.max(pred_ms['part'][idx_part].cpu(), dim=1)
                pred_ms['part'][idx_part] = p_max.squeeze(0)
            pred_ms = as_numpy(pred_ms)

        # calculate accuracy and SEND THEM TO MASTER
        result_queue.put_nowait(get_metrics(pred_ms, data_np))

        # visualization
        # if args.visualize:
        #     visualize_result(
        #         (data['img_ori'], seg_label, data['info']),
        #         preds, args)


def worker(args, dev_id, start_idx, end_idx, result_queue):
    torch.cuda.set_device(dev_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        broden_dataset.record_list['validation'], args,
        max_sample=args.num_val, start_idx=start_idx,
        end_idx=end_idx)
    loader_val = torchdata.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

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

    # Main loop
    evaluate(segmentation_module, loader_val, args, dev_id, result_queue)


def get_benchmark_result(result):
    assert len(result) == len(broden_dataset.record_list['validation'])

    benchmark = {k: {} for k in ['object', 'part', 'scene', 'material']}

    # object
    object_pixel = sum([item['object']['pixel'] for item in result if item['valid_object']])
    object_acc = sum([item['object']['acc'] for item in result if item['valid_object']])
    object_inter = sum([item['object']['inter'] for item in result if item['valid_object']])
    object_uni = sum([item['object']['uni'] for item in result if item['valid_object']])
    benchmark['object']['pixel_acc'] = object_acc / float(object_pixel)
    benchmark['object']['mIoU'] = (object_inter / (object_uni + 1e-8)).mean()

    # part
    mIoU_part, pixel_acc_part = [], []
    for object_part_idx, object_label in enumerate(broden_dataset.object_with_part):
        part_pixel_cnt, part_acc_cnt = 0, 0
        part_inter_cnt, part_uni_cnt = 0, 0
        for item in result:
            if not (item['valid_object'] and item["valid_part"][object_part_idx]):
                continue
            part_pixel_cnt += item["part"][object_part_idx]["pixel"]
            part_acc_cnt += item["part"][object_part_idx]["acc"]
            part_inter_cnt += item["part"][object_part_idx]["inter"]
            part_uni_cnt += item["part"][object_part_idx]["uni"]

        # report object not in validations set
        if part_pixel_cnt == 0:
            print("{}({}), no valid parts found in val set.".format(
                object_label, broden_dataset.names['object'][object_label]))
            continue

        mIoU_part.append((part_inter_cnt / (part_uni_cnt + 1e-8)).mean())
        pixel_acc_part.append(part_acc_cnt / float(part_pixel_cnt))

    benchmark['part']['pixel_acc'] = np.mean(pixel_acc_part)
    benchmark['part']['mIoU'] = np.mean(mIoU_part)

    # scene
    benchmark['scene']['top1'] = np.mean(
        [item['scene']['top1'] for item in result if item['scene']['gt'] != -1])

    # material
    material_pixel = sum([item['material']['pixel'] for item in result if item['valid_material']])
    material_acc = sum([item['material']['acc'] for item in result if item['valid_material']])
    material_inter = sum([item['material']['inter'] for item in result if item['valid_material']])
    material_uni = sum([item['material']['uni'] for item in result if item['valid_material']])
    benchmark['material']['pixel_acc'] = material_acc / float(material_pixel)
    benchmark['material']['mIoU'] = (material_inter / (material_uni + 1e-8)).mean()

    return benchmark


def main(args):
    # Parse device ids
    default_dev, *parallel_dev = parse_devices(args.devices)
    all_devs = parallel_dev + [default_dev]
    all_devs = [int(x.replace('gpu', '')) for x in all_devs]
    nr_devs = len(all_devs)

    print("nr_dev: {}".format(nr_devs))

    nr_files = len(broden_dataset.record_list['validation'])
    if args.num_val > 0:
        nr_files = min(nr_files, args.num_val)
    nr_files_per_dev = math.ceil(nr_files / nr_devs)

    pbar = tqdm(total=nr_files)

    result_queue = Queue(500)
    procs = []
    for dev_id in range(nr_devs):
        start_idx = dev_id * nr_files_per_dev
        end_idx = min(start_idx + nr_files_per_dev, nr_files)
        proc = Process(target=worker, args=(args, dev_id, start_idx, end_idx, result_queue))
        print('process:%d, start_idx:%d, end_idx:%d' % (dev_id, start_idx, end_idx))
        proc.start()
        procs.append(proc)

    # master fetches results
    processed_counter, all_result = 0, []
    while processed_counter < nr_files:
        if result_queue.empty():
            continue
        all_result.append(result_queue.get())
        processed_counter += 1
        pbar.update(1)

    for p in procs:
        p.join()

    benchmark = get_benchmark_result(all_result)

    print('[Eval Summary]:')
    print(benchmark)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', required=True,
                        help="a name for identifying the model to load")
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--list_val',
                        default='./data/validation.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[450], nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g.  300 400 500 600')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')

    # Misc arguments
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--visualize', action='store_true',
                        help='output visualization?')
    parser.add_argument('--result', default='./result',
                        help='folder to output visualization results')
    parser.add_argument('--devices', default='gpu0',
                        help='gpu_id for evaluation')

    args = parser.parse_args()
    print(args)

    nr_classes = broden_dataset.nr.copy()
    nr_classes['part'] = sum(
        [len(parts) for obj, parts in broden_dataset.object_part.items()])
    args.nr_classes = nr_classes

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.ckpt, args.id,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.ckpt, args.id,
                                        'decoder' + args.suffix)
    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
