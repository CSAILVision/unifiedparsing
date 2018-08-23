#!/usr/bin/env mdl 
import os
import signal
from csv import DictReader
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import numpy
from scipy.misc import imread

from broden_dataset.loadseg import AbstractSegmentation


class OpenSurfaceSegmentation(AbstractSegmentation):
    def __init__(self, directory):
        directory = os.path.expanduser(directory)
        self.directory = directory
        # Process material labels: open label-substance-colors.csv
        subst_name_map = {}
        with open(os.path.join(directory, 'label-substance-colors.csv')) as f:
            for row in DictReader(f):
                subst_name_map[row['substance_name']] = int(row['red_color'])
        # NOTE: substance names should be normalized. 
        self.substance_names = ['-'] * (1 + max(subst_name_map.values()))
        for k, v in list(subst_name_map.items()):
            self.substance_names[v] = k
        # Now load the metadata about images from photos.csv
        with open(os.path.join(directory, 'photos.csv')) as f:
            self.image_meta = list(DictReader(f))
            scenes = set(row['scene_category_name'] for row in self.image_meta)

    def all_names(self, category, j):
        if j == 0:
            return []
        if category == 'material':
            return [norm_name(n) for n in self.substance_names[j].split('/')]
        return []

    def size(self):
        """Returns the number of images in this dataset."""
        return len(self.image_meta)

    def filename(self, i):
        """Returns the filename for the nth dataset image."""
        photo_id = int(self.image_meta[i]['photo_id'])
        return os.path.join(self.directory, 'photos', '%d_resized.jpg' % photo_id)

    def metadata(self, i):
        """Returns an object that can be used to create all segmentations."""
        row = self.image_meta[i]
        return dict(
            filename=self.filename(i),
            seg_filename=self.seg_filename(i))

    def seg_filename(self, i):
        """ Return the seg filename for the nth dataset seg img. """
        photo_id = int(self.image_meta[i]['photo_id'])
        return os.path.join(self.directory, 'photos-labels', '%d_resized.png' % photo_id)

    @classmethod
    def resolve_segmentation(cls, m, categories=None):
        result = {}
        if wants('material', categories):
            labels = imread(m['seg_filename'])
            result['material'] = labels[:, :, 0]
        arrs = [a for a in list(result.values()) if len(numpy.shape(a)) >= 2]
        shape = arrs[0].shape[-2:] if arrs else (1, 1)
        return result, shape


def norm_name(s):
    return s.replace(' - ', '-').replace('/', '-').strip().lower()


def wants(what, option):
    if option is None:
        return True
    return what in option


def generate_resized_os(os_dir):
    """
    resize img and material. 
    all samples are resized, 25352. 
    """
    ds = OpenSurfaceSegmentation(directory=os_dir)
    all_result = map_in_pool(partial(resize_data, verbose=True),
                             all_dataset_segmentations({"opensurfaces": ds}),
                             single_process=False,
                             verbose=True)


def resize_data(record, verbose):
    """
    resize data in record. 
    """
    dataset, file_index, filename, md = record
    img_path, seg_path = md['filename'], md['seg_filename']
    if verbose:
        print("{} {}".format(file_index, os.path.basename(img_path)))
    # short size maximum 512 as ade challenge
    # original code seem to have bug 
    img = cv2.imread(img_path)
    h, w = img.shape[0], img.shape[1]
    h_new, w_new = 0, 0
    max_size = 512
    if h <= w and h > max_size:
        h_new, w_new = max_size, round(w / float(h) * max_size)
    elif w <= h and w > max_size:
        h_new, w_new = round(h / float(w) * max_size), max_size
    else:
        return 0
    cv2.imwrite(img_path, cv2.resize(img, (w_new, h_new),
                                     interpolation=cv2.INTER_LINEAR))
    # resize obj seg 
    seg = cv2.imread(seg_path)
    cv2.imwrite(seg_path, cv2.resize(seg, (w_new, h_new),
                                     interpolation=cv2.INTER_NEAREST))
    return 0


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
        return res.get(31536000)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()


def all_dataset_segmentations(data_sets):
    for name, ds in list(data_sets.items()):
        for i in list(range(ds.size())):
            yield (name, i, ds.filename(i), ds.metadata(i))


def setup_sigint():
    return signal.signal(signal.SIGINT, signal.SIG_IGN)


def restore_sigint(original):
    signal.signal(signal.SIGINT, original)

