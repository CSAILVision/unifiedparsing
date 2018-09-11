
import json
import os

import numpy

from .loadseg import AbstractSegmentation


class DtdSegmentation(AbstractSegmentation):
    def __init__(self, directory):
        directory = os.path.expanduser(directory)
        self.directory = directory
        with open(os.path.join(directory, 'labels',
                               'labels_joint_anno.txt')) as f:
            self.dtd_meta = [line.split(None, 1) for line in f.readlines()]
        # do not include '-' in texture names. No unlabeled sample in data. 
        self.textures = sorted(list(set(sum(
            [c.split() for f, c in self.dtd_meta], []))))
        self.texture_map = dict((t, i) for i, t in enumerate(self.textures))

    def all_names(self, category, j):
        if j == 0:
            return []
        if category == 'texture':
            return [self.textures[j]]
        return []

    def size(self):
        """Returns the number of images in this dataset."""
        return len(self.dtd_meta)

    def filename(self, i):
        """Returns the filename for the nth dataset image."""
        return os.path.join(self.directory, 'images', self.dtd_meta[i][0])

    def metadata(self, i):
        """Returns an object that can be used to create all segmentations."""
        fn, tnames = self.dtd_meta[i]
        # parse main label name
        main_label_name = fn.split('/')[0]
        # tnumbers = [self.texture_map[n] for n in tnames.split()]
        tnumbers = [self.texture_map[main_label_name]]
        return dict(
            filename=os.path.join(self.directory, 'images', fn),
            tnumbers=tnumbers
        )

    @classmethod
    def resolve_segmentation(cls, m, categories=None):
        filename, tnumbers = m["filename"], m["tnumbers"]
        result = {}
        if wants('texture', categories):
            result['texture'] = tnumbers
        arrs = [a for a in list(result.values()) if len(numpy.shape(a)) >= 2]
        shape = arrs[0].shape[-2:] if arrs else (1, 1)
        return result, shape

    def training_records(self):
        record_path = os.path.join("./meta_file", "dtd_training_list.json")
        with open(record_path) as f:
            filelist_json = f.readlines()
        return [json.loads(x) for x in filelist_json]

    def validation_records(self):
        record_path = os.path.join("./meta_file", "dtd_validation_list.json")
        with open(record_path) as f:
            filelist_json = f.readlines()
        return [json.loads(x) for x in filelist_json]


def wants(what, option):
    if option is None:
        return True
    return what in option
