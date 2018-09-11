import numpy as np
import re
import functools
import cv2


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def MydrawMask(img, masks, lr=(None, None), alpha=None, clrs=None, info=None):
    n, h, w = masks.shape[0], masks.shape[1], masks.shape[2]
    if lr[0] is None:
        lr = (0, n)
    if alpha is None:
        alpha = [.4, .4, .4]
    alpha = [.6, .6, .6]
    if clrs is None:
        clrs = np.zeros((n,3)).astype(np.float)
        for i in range(n):
            for j in range(3):
                clrs[i][j] = np.random.random()*.6+.4

    for i in range(max(0, lr[0]), min(n, lr[1])):
        M = masks[i].reshape(-1)
        B = np.zeros(h*w, dtype = np.int8)
        ix, ax, iy, ay = 99999, 0, 99999, 0
        for y in range(h-1):
            for x in range(w-1):
                k = y*w+x
                if M[k] == 1:
                    ix = min(ix, x)
                    ax = max(ax, x)
                    iy = min(iy, y)
                    ay = max(ay, y)
                if M[k] != M[k+1]:
                    B[k], B[k+1] =1,1
                if M[k] != M[k+w]:
                    B[k], B[k+w] =1,1
                if M[k] != M[k+1+w]:
                    B[k], B[k+1+w] = 1,1
        M.shape = (h,w)
        B.shape = (h,w)
        for j in range(3):
            O,c,a = img[:,:,j], clrs[i][j], alpha[j]
            am = a*M
            O = O - O*am + c*am*255
            img[:,:,j] = O*(1-B)+c*B
        #cv2.rectangle(img, (ix,iy), (ax,ay), (0,255,0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = ix-1, iy-1
        if x<0:
            x=0
        if y<10:
            y+=7
        if int(img[y,x,0])+int(img[y,x,1])+int(img[y,x,2]) > 650:
            col = (255,0,0)
        else:
            col = (255,255,255)
        #col = (255,0,0)
        #cv2.putText(img, id2class[info['category_id']]+': %.3f' % info['score'], (x, y), font, .3, col, 1)
    return img


def maskrcnn_colorencode(img, label_map, color_list):
    # do not modify original list
    label_map = np.array(np.expand_dims(label_map, axis=0), np.uint8)
    #label_map = label_map.transpose(1, 2, 0)
    label_list = list(np.unique(label_map))
    out_img = img.copy()
    for i, label in enumerate(label_list):
        if label == 0: continue
        this_label_map = (label_map == label)
        alpha = [0, 0, 0]
        o = i
        if o >= 6:
            o = np.random.randint(1, 6)
        o_lst = [o%2, (o // 2)%2, o//4]
        for j in range(3):
            alpha[j] = np.random.random() * 0.5 + 0.45
            alpha[j] *= o_lst[j]
        out_img = MydrawMask(out_img, this_label_map, alpha=alpha,
                clrs=np.expand_dims(color_list[label], axis=0))
    return out_img


def remove_small_mat(seg_mat, seg_obj, threshold=0.1):
    object_list = np.unique(seg_obj)
    seg_mat_new = np.zeros_like(seg_mat)
    for obj_label in object_list:
        obj_mask = (seg_obj == obj_label)
        mat_result = seg_mat * obj_mask
        mat_sum = obj_mask.sum()
        for mat_label in np.unique(mat_result):
            mat_area = (mat_result == mat_label).sum()
            if mat_area / float(mat_sum) < threshold:
                continue
            seg_mat_new += mat_result * (mat_result == mat_label)
        # sorted_mat_index = np.argsort(-np.asarray(mat_area))
    return seg_mat_new


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


def intersection_union_part(pred, gt, nr_classes):
    # nr_classes include background 0.
    # Compute area intersection without 0:
    (area_intersection, _) = np.histogram(pred * (gt == pred),
            bins=nr_classes - 1, range=(1, nr_classes - 1))
    # Compute area union without 0:
    (area_pred, _) = np.histogram(pred, bins=nr_classes - 1, range=(1, nr_classes - 1))
    (area_lab, _) = np.histogram(gt, bins=nr_classes - 1, range=(1, nr_classes - 1))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):
    
    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                    'Can not recognize device: "%s"' % d)
    return ret
