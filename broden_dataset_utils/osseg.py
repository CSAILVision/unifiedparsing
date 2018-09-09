
import os
from csv import DictReader

import numpy
from scipy.misc import imread

from broden_dataset_utils.loadseg import AbstractSegmentation


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
