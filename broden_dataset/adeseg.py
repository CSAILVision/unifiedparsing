#!/usr/bin/env mdl
import glob
import os
import re
import signal
from collections import namedtuple
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import numpy
from scipy.io import loadmat
from scipy.misc import imread

from broden_dataset.loadseg import AbstractSegmentation
from config import config


class AdeSegmentation(AbstractSegmentation):
    def __init__(self, directory, version):
        # Default to value of ADE20_ROOT env variable
        if directory is None:
            directory = os.environ['ADE20K_ROOT']
        directory = os.path.expanduser(directory)
        # Default to the latest version present in the directory
        if version is None:
            contents = os.listdir(directory)
            if not list(c for c in contents if re.match('^index.*mat$', c)):
                version = sorted(c for c in contents if os.path.isdir(
                    os.path.join(directory, c)))[-1]
            else:
                version = ''
        self.root = directory
        self.version = version
        mat = loadmat(self.expand(self.version, 'index*.mat'), squeeze_me=True)
        index = mat['index']
        Ade20kIndex = namedtuple('Ade20kIndex', index.dtype.names)
        self.index = Ade20kIndex(
            **{name: index[name][()] for name in index.dtype.names})
        # Here we use adechallenger scene label instead of ade20k.
        with open(config.scene_label_path, 'r') as f:
            lines = f.readlines()
        self.index_scene_adecha = []
        for i, l in enumerate(lines):
            l = l.split(" ")
            filename, scene_label = l[0], l[1].replace('\n', '')
            if scene_label == 'misc':
                scene_label = '-'
            assert filename + '.jpg' == self.index.filename[i]
            self.index_scene_adecha.append(scene_label)
        self.scenes = ['-'] + [s for s in sorted(set(self.index_scene_adecha)) if s != '-']
        self.scene_map = dict((s, i) for i, s in enumerate(self.scenes))

    def all_names(self, category, j):
        if j == 0:
            return []
        if category == 'scene':
            return [self.scenes[j] + '-s']
        result = self.index.objectnames[j - 1]
        return re.split(',\s*', result)

    def size(self):
        """Returns the number of images in this dataset."""
        return len(self.index.filename)

    def filename(self, n):
        """Returns the filename for the nth dataset image."""
        filename = self.index.filename[n]
        # if self.use_resized and self.resized_file_flag[n]:
        #     filename = re.sub(r'\.jpg$', '_resized.jpg', filename)
        folder = self.index.folder[n]
        return self.expand(folder, filename)

    def metadata(self, i):
        """Returns an object that can be used to create all segmentations."""
        return dict(
            filename=self.filename(i),
            seg_filename=self.seg_filename(i),
            part_filenames=self.part_filenames(i),
            scene=self.scene_map[self.index_scene_adecha[i]]
        )

    @classmethod
    def resolve_segmentation(cls, m, categories=None):
        # raise NotImplementedError("resize images")
        result = {}
        if wants('scene', categories):
            result['scene'] = m['scene']
        if wants('part', categories):
            result['part'] = load_parts(m)
        if wants('object', categories):
            result['object'] = load_segmentation(m)
        arrs = [a for a in list(result.values()) if len(numpy.shape(a)) >= 2]
        shape = arrs[0].shape[-2:] if arrs else (1, 1)
        return result, shape

    ### End of contract for AbstractSegmentation

    def seg_filename(self, n):
        """Returns the segmentation filename for the nth dataset image."""
        return re.sub(r'\.jpg$', '_seg.png', self.filename(n))

    def part_filenames(self, n):
        """Returns all the subpart images for the nth dataset image."""
        filename = self.filename(n)
        result = []
        probe = re.sub(r'\.jpg$', '_parts_1.png', filename)
        if os.path.isfile(probe):
            result.append(probe)
        return result

    def expand(self, *path):
        """Expands a filename and directories with the ADE dataset"""
        result = os.path.join(self.root, *path)
        if '*' in result or '?' in result:
            globbed = glob.glob(result)
            if len(globbed):
                return globbed[0]
        return result

    def generate_train_val_record(self):
        raise NotImplementedError

    def training_records(self):
        raise NotImplementedError

    def validation_records(self):
        raise NotImplementedError


def norm_name(s):
    return s.replace(' - ', '-').replace('/', '-')


def load_segmentation(m):
    """Returns the nth dataset segmentation as a numpy array,
    where each entry at a pixel is an object class value.
    """
    data = imread(m['seg_filename'])
    return decodeClassMask(data)


def load_parts(m):
    """Returns an list of part segmentations for the nth dataset item,
    with one array for each level available.
    """
    result = []
    for fn in m['part_filenames']:
        data = imread(fn)
        result.append(decodeClassMask(data))
    if not result:
        return []
    return numpy.concatenate(tuple(m[numpy.newaxis] for m in result))


def decodeClassMask(im):
    """Decodes pixel-level object/part class and instance data from
    the given image, previously encoded into RGB channels."""
    # Classes are a combination of RG channels (dividing R by 10)
    return (im[:, :, 0] // 10) * 256 + im[:, :, 1]


def wants(what, option):
    if option is None:
        return True
    return what in option


def generate_resized_ade20k(ade_dir):
    ds = AdeSegmentation(
        directory=ade_dir,
        version="ADE20K_2016_07_26")
    data_sets = {"ade20k": ds}
    all_result = map_in_pool(partial(resize_data,
                                     verbose=True),
                             all_dataset_segmentations(data_sets),
                             single_process=False,
                             verbose=True)


def resize_data(record, verbose):
    """
    resize data in record. 
    """
    dataset, file_index, img_path, md = record
    if verbose:
        print("{} {}".format(file_index, os.path.basename(img_path)))
    # short size maximum 512 as ade challenge
    # original code seem to have bug 
    img = cv2.imread(img_path)
    h, w = img.shape[0], img.shape[1]
    h_new, w_new = 0, 0
    max_size = 512
    if w >= h > max_size:
        h_new, w_new = max_size, round(w / float(h) * max_size)
    elif h >= w > max_size:
        h_new, w_new = round(h / float(w) * max_size), max_size
    else:
        return 0
    cv2.imwrite(img_path, cv2.resize(img, (h_new, w_new),
                                     interpolation=cv2.INTER_LINEAR))
    # resize obj seg 
    seg_obj_path = md["seg_filename"]
    seg = cv2.imread(seg_obj_path)
    cv2.imwrite(seg_obj_path, cv2.resize(seg, (h_new, w_new),
                                         interpolation=cv2.INTER_NEAREST))
    # resize part seg 
    for i, seg_part_path in enumerate(md["part_filenames"]):
        seg = cv2.imread(seg_part_path)
        cv2.imwrite(seg_part_path, cv2.resize(seg, (h_new, w_new),
                                              interpolation=cv2.INTER_NEAREST))
    return 0


def all_dataset_segmentations(data_sets):
    for name, ds in list(data_sets.items()):
        for i in list(range(ds.size())):
            yield (name, i, ds.filename(i), ds.metadata(i))


def map_in_pool(fn, data, single_process=False, verbose=False):
    """
    Our multiprocessing solution; wrapped to stop on ctrl-C well.
    """
    if single_process:
        return list(map(fn, data))
    n_procs = min(cpu_count(), 32)
    original_sigint_handler = setup_sigint()
    pool = Pool(processes=n_procs, initializer=setup_sigint)
    restore_sigint(original_sigint_handler)
    try:
        if verbose:
            print('Mapping with %d processes' % n_procs)
        res = pool.map_async(fn, data)
        # what is this magic number means?
        return res.get(31536000)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()


def setup_sigint():
    return signal.signal(signal.SIGINT, signal.SIG_IGN)


def restore_sigint(original):
    signal.signal(signal.SIGINT, original)
