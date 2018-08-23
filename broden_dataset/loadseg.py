import cv2
import numpy


class AbstractSegmentation:
    def all_names(self, category, j):
        raise NotImplementedError

    def size(self, split=None):
        return 0

    def filename(self, i):
        raise NotImplementedError

    def metadata(self, i):
        return self.filename(i)

    @classmethod
    def resolve_segmentation(cls, m):
        return {}

    def name(self, category, i):
        """
        Default implementation for segmentation_data,
        utilizing all_names.
        """
        all_names = self.all_names(category, i)
        return all_names[0] if len(all_names) else ''

    def segmentation_data(self, category, i, c=0, full=False):
        """
        Default implementation for segmentation_data,
        utilizing metadata and resolve_segmentation.
        """
        segs, segs_shape = self.resolve_segmentation(
            self.metadata(i), categories=[category])
        if category not in segs:
            return 0
        data = numpy.asarray(segs[category])
        if not full and len(data.shape) >= 3:
            return data[0]
        return data

    def image_data(self, i):
        return cv2.imread(self.filename(i))

